print("CNN file started")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import os

# -----------------------------
# SETTINGS
# -----------------------------
DATASET_PATH = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = 128
MODEL_SAVE_PATH = "models/cnn_model.pth"

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

class_names = dataset.classes
print("Classes:", class_names)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Train: {train_size}")
print(f"Validation: {val_size}")
print(f"Test: {test_size}")

# -----------------------------
# CNN MODEL
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cpu")
print("Using device:", device)

# -----------------------------
# MODEL SETUP
# -----------------------------
model = SimpleCNN(len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAINING
# -----------------------------
print("Training CNN...")

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Model saved:", MODEL_SAVE_PATH)

# -----------------------------
# TEST MODEL
# -----------------------------
print("Testing model...")

model.eval()

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total

print(f"\nCNN Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))