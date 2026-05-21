import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kagglehub

DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

def download_dataset():
    path = kagglehub.dataset_download(DATASET_NAME)
    return Path(path)

def find_chest_xray_folder(dataset_root):
    dataset_root = Path(dataset_root)
    for folder in dataset_root.rglob("*"):
        normal_path = folder / "train" / "NORMAL"
        pneumonia_path = folder / "train" / "PNEUMONIA"
        if normal_path.exists() and pneumonia_path.exists():
            return folder
    raise FileNotFoundError("Dataset folder not found.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CONFIGURARE CORECTĂ: Folosim 224x224 pentru a se potrivi perfect cu aplicația Flask
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset_root = download_dataset()
chest_xray_folder = find_chest_xray_folder(dataset_root)
train_folder = chest_xray_folder / "train"

dataset = datasets.ImageFolder(train_folder, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# CONFIGURARE CORECTĂ: Modelul adaptat pentru 224x224 (intrare de 100.352)
class UltraLightCNN(nn.Module):
    def __init__(self):
        super(UltraLightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Dimensiunea devine 112x112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Dimensiunea devine 56x56
        )
        # 32 de canale * 56 * 56 = 100352 trăsături
        self.classifier = nn.Linear(32 * 56 * 56, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = UltraLightCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx + 1}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"--- Final Epocă {epoch+1}/{EPOCHS} - Loss Mediu: {running_loss / len(loader):.4f} ---\n")

torch.save(model.state_dict(), "classifier_model.pth")
print("\nClassifier saved nou: classifier_model.pth")