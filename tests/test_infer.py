# Unit test placeholder"""
Inference script for DeepFake Detector.
"""
import torch
import cv2
import numpy as np
from src.models.resnext_lite import ResNeXtLite

def predict_from_video(video_path: str, checkpoint: str = "checkpoints/model_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNeXtLite().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
        frames.append(frame)
        if len(frames) >= 4:  # sample 4 frames
            break
    cap.release()

    x = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch.sigmoid(model(x)).item()
    label = "FAKE" if out > 0.5 else "REAL"
    return out, label

if __name__ == "__main__":
    score, label = predict_from_video("data/sample/fake1.mp4")
    print(f"Label: {label}, Score: {score:.3f}")
