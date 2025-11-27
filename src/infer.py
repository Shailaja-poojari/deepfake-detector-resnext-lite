import torch
import cv2
import numpy as np
from torchvision import transforms
from src.models.resnext_lite import ResNeXtLite

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model inside HuggingFace app externally
def preprocess(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def predict_frame(frame, model):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img).to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0][1].item() * 100

    return round(prob, 2)
