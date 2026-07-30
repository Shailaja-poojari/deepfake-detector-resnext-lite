# DeepFake Detection System - ResNeXt Lite + Temporal Modeling

A lightweight and efficient deepfake detection system designed for high-speed inference and real-world deployment scenarios.

This project leverages a ResNeXt-based architecture for spatial feature extraction combined with temporal consistency analysis to detect manipulated media.

---

##  Overview

Deepfake media poses a growing threat in digital ecosystems, including misinformation, fraud, and identity misuse.

This project aims to build a **scalable, efficient, and accurate deepfake detection pipeline** that can:

- Detect manipulated videos/images
- Maintain high accuracy with reduced computational cost
- Support real-time or near real-time inference

---

##  Key Features

-  **ResNeXt Lite Backbone** for efficient feature extraction
-  **Temporal Analysis Module** for detecting frame inconsistencies
-  **Optimized for Fast Inference** (lightweight architecture)
-  **Modular and scalable project structure**
-  **Docker-ready setup for deployment**
-  **Training and evaluation pipelines included**

---

##  Architecture

- **Input** → Video / Image frames  
- **Preprocessing** → Frame extraction + normalization  
- **Model** → ResNeXt Lite (CNN backbone)  
- **Temporal Head** → Sequence consistency analysis  
- **Output** → Real / Fake classification  

---

##  Tech Stack

- Python
- PyTorch
- NumPy / OpenCV
- Jupyter Notebook (for experimentation)
- Docker (for deployment setup)

---

## Project Structure

```text
deepfake-detector-resnext-lite/
├── app/                 # Inference application
├── src/                 # Core model architecture and utilities
├── notebooks/           # Training and experimentation notebooks
├── tests/               # Unit and inference tests
├── docker/              # Docker configuration files
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment configuration
└── app.py               # Application entry point
```

---

##  Model Training

- Training conducted using **Google Colab (GPU environment)**
- Dataset preprocessed with frame extraction techniques
- Applied data augmentation to improve generalization
- Optimized hyperparameters for performance vs efficiency trade-off

---

##  Results

- Achieved ~90%+ classification accuracy on validation data
- Reduced model size compared to standard CNN architectures
- Improved inference speed suitable for real-time applications

---

##  Inference

Run locally:

```bash
pip install -r requirements.txt
python app.py

---

🐳 Docker Setup
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector

---

Challenges & Improvements

Challenges:
Handling temporal inconsistencies across frames
Balancing model accuracy with computational efficiency
Dataset quality and variability

---

Future Improvements:
Integrate transformer-based architectures
Deploy via Hugging Face / cloud APIs
Improve robustness on low-quality videos

---

Real-World Use Cases:
Social media content verification
Fraud detection in video-based systems
Media authenticity validation platforms

---

Author
Shailaja Poojary
