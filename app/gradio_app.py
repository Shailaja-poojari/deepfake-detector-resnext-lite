import gradio as gr
import cv2
from src.infer import predict_frame
from src.extra.audio_visual_check import av_consistency
from src.extra.report_generator import generate_pdf, generate_csv
from src.extra.authentication import signup, login
import torch


model = torch.load("model.pth", map_location="cpu")
model.eval()

def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    score = predict_frame(frames[0], model)
    consistency = av_consistency(frames, video_path)

    gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_path = "heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap)

    pdf = generate_pdf(score, consistency, heatmap_path)
    csv = generate_csv(score, consistency)

    return score, consistency, heatmap_path, pdf, csv


def ui():
    with gr.Blocks() as demo:

        with gr.Tab("Login"):
            u = gr.Textbox(label="Username")
            p = gr.Textbox(label="Password", type="password")
            btn = gr.Button("Login")
            out = gr.Textbox()
            btn.click(login, [u,p], out)

        with gr.Tab("Signup"):
            u2 = gr.Textbox(label="New Username")
            p2 = gr.Textbox(label="New Password", type="password")
            btn2 = gr.Button("Create Account")
            out2 = gr.Textbox()
            btn2.click(signup, [u2,p2], out2)

        with gr.Tab("Deepfake Detection"):
            video = gr.Video()
            run = gr.Button("Analyze")

            score = gr.Number(label="Fake Probability (%)")
            cons = gr.Number(label="AV Consistency")
            heat = gr.Image(label="Manipulated Regions")
            pdf = gr.File()
            csv = gr.File()

            run.click(analyze, [video], [score, cons, heat, pdf, csv])

    return demo

app = ui()

if __name__ == "__main__":
    app.launch()
