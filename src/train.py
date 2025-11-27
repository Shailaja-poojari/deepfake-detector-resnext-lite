import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
from src.models.resnext_lite import ResNeXtLite

# -------------------------------------------------------
# SIMPLE CUSTOM DATASET (images in real/ and fake/ folder)
# -------------------------------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, root_folder):
        self.items = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

        real_path = os.path.join(root_folder, "real")
        fake_path = os.path.join(root_folder, "fake")

        for img in os.listdir(real_path):
            self.items.append((os.path.join(real_path, img), 0))

        for img in os.listdir(fake_path):
            self.items.append((os.path.join(fake_path, img), 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label)


# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_model(model, epochs, lr, batch_size, save_path, data_path="dataset"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    dataset = DeepfakeDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss = {total_loss:.4f}")

    torch.save(model, save_path)
    print("Saved model to:", save_path)

    return save_path
