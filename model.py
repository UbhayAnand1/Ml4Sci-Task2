import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class YourDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Set the path to the CSV file containing the labels
        csv_path = os.path.join(data_dir, 'classifications.csv')

        # Load the labels from the CSV file
        self.labels = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the image ID and label for the current index
        image_id = int(self.labels.iloc[idx]['ID'])
        label = self.labels.iloc[idx]['is_lens']

        # Set the path to the image file
        image_path = os.path.join(self.data_dir, 'jpeg_files', f'imageEUC_VIS-{image_id}.jpg')

        # Load the image
        image = Image.open(image_path)

        # Apply any transforms to the image
        if self.transform:
            image = self.transform(image)

        return image, label

# Set the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the batch size
batch_size = 32

# Set the number of epochs to train for
num_epochs = 9

# Set the path to the directory containing your data
data_dir = r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining'

# Define any transforms that you want to apply to the images
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

# Create a Dataset object for your training data
train_dataset = YourDataset(data_dir, transform=transform)

# Create a DataLoader for your training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load a pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Unfreeze all layers of the model
for param in model.parameters():
    param.requires_grad = True

# Move the model to the device
model = model.to(device)

# Set the loss function
criterion = nn.CrossEntropyLoss()

# Set the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0009, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Convert the labels to a Long tensor
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(epoch+1, train_loss, train_acc))

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')