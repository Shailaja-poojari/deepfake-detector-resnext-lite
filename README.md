# DeepFake Detector – ResNeXt Lite
AI/ML system to detect face-swap deepfake videos using ResNeXt-Lite and CNN-based classification.

## Phases
1. **Training** – run `src/train.py` (in Colab).
2. **Inference** – run `infer.py` on test videos.
3. **Gradio Demo** – run `app/gradio_app.py`.
4. **Deployment** – push to Hugging Face Spaces.

## Model
- Backbone: ResNeXt-50 (Lite)
- Input: 224x224 face crops
- Output: Real / Fake
