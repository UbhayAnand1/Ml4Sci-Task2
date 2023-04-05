
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# Set the device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a custom dataset
class LensDataset(Dataset):
    def __init__(self, csv_file, jpg_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.jpg_dir = jpg_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    # Get the file name and label for the current sample
        filename = 'imageEUC_VIS-' + str(int(self.df.iloc[idx]['ID'])) + '.jpg'
        label = torch.tensor(self.df.iloc[idx]['is_lens'], dtype=torch.long)

    # Load the image
        image = Image.open(os.path.join(self.jpg_dir, filename))

    # Apply the transformation (if any)
        if self.transform:
            image = self.transform(image)

        return image, label

# Create the train and test datasets
train_dataset = LensDataset(
    csv_file=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\train.csv',
    jpg_dir=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\jpg_files',
    transform=data_transform
)
test_dataset = LensDataset(
    csv_file=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\test.csv',
    jpg_dir=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\jpg_files',
    transform=data_transform
)

# Create the data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Replace the last fully connected layer with a new one with 2 outputs (for binary classification)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Move the model to the device
model.to(device)

# Set the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Fine-tune the model on the training data
num_epochs = 4
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the running loss and corrects
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Print the average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(epoch+1, epoch_loss, epoch_acc))

# Evaluate the model on the test data
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update the number of correctly classified samples
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the accuracy on the test data
accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')