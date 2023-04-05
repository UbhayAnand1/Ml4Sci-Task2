from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
import os


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

# Set the device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
state_dict = torch.load(r'C:\Users\abhay\OneDrive\Desktop\task2\trained_model.pth')

# Load the state dictionary into the new model object
model.load_state_dict(state_dict)

# Move the model to the device
model.to(device)

# Set the model to evaluation mode
model.eval()   
batch_size = 32

# Create the test dataset and data loader
test_dataset = LensDataset(
    csv_file=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\test.csv',
    jpg_dir=r'C:\Users\abhay\OneDrive\Desktop\task2\SpaceBasedTraining\jpg_files',
    transform=data_transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize lists to store the true labels and predicted probabilities
true_labels = []
predicted_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Update the lists of true labels and predicted probabilities
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(probs[:, 1].cpu().numpy())

# Compute the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()