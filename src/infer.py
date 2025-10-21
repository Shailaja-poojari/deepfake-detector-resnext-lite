import torch, torchvision.transforms as transforms, cv2, numpy as np
from src.models.resnext_lite import ResNeXtLite

def load_model(weights="checkpoints/model_best.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNeXtLite(num_classes=2)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    return model, device

def preprocess_frame(frame):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return t(frame).unsqueeze(0)

def predict_video(video_path):
    model, device = load_model()
    cap = cv2.VideoCapture(video_path)
    preds=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        x=preprocess_frame(frame).to(device)
        with torch.no_grad():
            prob=torch.softmax(model(x),dim=1)
        preds.append(prob.cpu().numpy())
    cap.release()
    avg=np.mean(preds,axis=0)
    label="Fake" if np.argmax(avg)==1 else "Real"
    conf=float(np.max(avg))
    return {label: conf}
