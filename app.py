import gradio as gr
from src.infer import predict_video

def analyze(video):
    if video is None:
        return "Please upload a video."
    result = predict_video(video)
    label = list(result.keys())[0]
    confidence = round(result[label], 3)
    return f"Prediction: {label} (confidence: {confidence})"

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Video(label="Upload a video"),
    outputs="text",
    title="DeepFake Detector - ResNeXt Lite",
    description="Upload a video to check if it is REAL or FAKE using the ResNeXt-Lite deepfake classifier.",
)

if __name__ == "__main__":
    demo.launch()
