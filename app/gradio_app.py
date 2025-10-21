import gradio as gr
from infer import predict_video

def analyze(video):
    return predict_video(video)

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Video(label="Upload a short video"),
    outputs=gr.Label(label="Prediction (Real or Fake)"),
    title="DeepFake Detector – ResNeXt Lite",
    description="AI-based detection of face-swap deepfakes using ResNeXt-Lite model."
)

if __name__ == "__main__":
    demo.launch()
