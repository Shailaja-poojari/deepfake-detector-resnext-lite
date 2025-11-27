import gradio as gr
import cv2
import torch
from src.infer import predict_frame
from src.extra.audio_visual_check import av_consistency
from src.extra.report_generator import generate_pdf, generate_csv

model = torch.load("model.pth", map_location="cpu")
model.eval()

def analyze(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    score = predict_frame(frames[0], model)
    consistency = av_consistency(frames, video)

    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heat_path = "heatmap.jpg"
    cv2.imwrite(heat_path, heat)

    pdf = generate_pdf(score, consistency, heat_path)
    csv = generate_csv(score, consistency)

    return score, consistency, heat_path, pdf, csv

with gr.Blocks() as demo:
    gr.Markdown("# Deepfake Detector (ResNeXt-Lite)")
    video = gr.Video()
    btn = gr.Button("Analyze")

    score = gr.Number()
    cons = gr.Number()
    heat = gr.Image()
    pdf = gr.File()
    csv = gr.File()

    btn.click(analyze, [video], [score, cons, heat, pdf, csv])

demo.launch()
