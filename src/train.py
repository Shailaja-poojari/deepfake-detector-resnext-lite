import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from models.resnext_lite import ResNeXtLite

# --- Dummy dataset class (replace with FaceForensics later) ---
class DummyDataset(Dataset):
    def __init__(self, root="data/train", transform=None):
        self.root, self.transform = root, transform
        self.samples = []
        for cls in ["real", "fake"]:
            folder = os.path.join(root, cls)
            if not os.path.exists(folder): continue
            for img in os.listdir(folder):
                if img.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(folder, img), 0 if cls=="real" else 1))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# --- Training loop ---
def train():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    dataset = DummyDataset(transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNeXtLite().to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        total, correct = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct/total*100
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, "checkpoints/model_best.pth")
    print("✅ Model saved at checkpoints/model_best.pth")

if __name__ == "__main__":
    train()
