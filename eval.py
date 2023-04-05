import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.fc = nn.Linear(self.resnet_model.fc.in_features, 3)

    def forward(self, x):
        x = self.resnet_model(x)
        return x


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
# Save only the state dictionary for the ResNet module
# Save only the state dictionary for the ResNet module
torch.save(model.model.state_dict(), r'C:\Users\abhay\OneDrive\Desktop\task2\trained_model.pth')

# Load only the state dictionary for the ResNet module
model = ResNet()
torch.save(model.model.state_dict(), r'C:\Users\abhay\OneDrive\Desktop\task2\trained_model.pth')

model = model.to(device)
model.eval()

# Rest of the code for data loading, prediction, ROC curve and AUC calculation


# Define the data transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test data
test_data = ImageFolder(root=r'C:\Users\abhay\OneDrive\Desktop\twask2\SpaceBasedTraining\jpeg.files', transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize lists to store the true labels and predicted probabilities
y_true = []
y_pred = []

# Iterate over the test data and make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(probabilities)

# Convert y_pred to a NumPy array
y_pred = np.array(y_pred)

# Compute the ROC curve and AUC score for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, y_pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'Class {i}: AUC = {roc_auc[i]:.4f}')
    
    # Plot the ROC curve
    plt.plot(fpr[i], tpr[i], label=f'Class {i}, AUC = {roc_auc[i]:.4f}')

# Plot the ROC curves for all classes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
