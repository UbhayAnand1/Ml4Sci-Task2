import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

    def forward(self, x):
        x = self.model(x)
        return x

# Define the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define the hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the datasets
train_dataset = datasets.ImageFolder(root='C:/Users/abhay/OneDrive/Desktop/task1/dataset_preprocessed/train',
                                      transform=transform)
test_dataset = datasets.ImageFolder(root='C:/Users/abhay/OneDrive/Desktop/task1/dataset_preprocessed/test',
                                     transform=transform)

# Define the data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and move it to the device
model = ResNet().to(device)

# Define the loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
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
import os
print(os.getcwd())
