import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = ResNet()
model.load_state_dict(torch.load(r'C:\Users\abhay\OneDrive\Desktop\Lens_classification\trained_model.pth'))
model = model.to(device)
model.eval()

# Define the data transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test data
test_data = ImageFolder(root=r'C:\Users\abhay\OneDrive\Desktop\Lens_classification\dataset_preprocessed\test', transform=data_transforms)
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

# TODO: Add code to plot the ROC curves and display the AUC scores for each class