# AI Interview Coach

A real-time, multi-modal AI system that analyzes interview performance through webcam and microphone. Four custom-trained deep learning models evaluate speech confidence, facial expressions, body posture, and eye contact — producing a final confidence score with actionable feedback.

---

## Demo

> *[Add GIF or screenshot of the live system here]*

---

## Features

- Real-time speech confidence analysis using CNN + LSTM on MFCC features
- Facial expression classification with custom CNN (trained from scratch on FER2013)
- Interview posture scoring using Neural Network on MediaPipe pose landmarks
- Eye contact detection using iris gaze geometry from MediaPipe Face Mesh
- Multi-modal score fusion → single confidence score (0-100) with grade
- Actionable text feedback after each session
- Full-stack: Flask API backend + React frontend

---

## Architecture

```
Webcam + Mic
    │
    ├── Audio → MFCC → Speech CNN/LSTM → Confidence Score (30%)
    ├── Face  → CNN → Expression Score (25%)
    ├── Pose  → MediaPipe → Posture NN → Posture Score (25%)
    └── Iris  → Face Mesh → Eye Contact Score (20%)
                          │
                   Score Fusion Layer
                          │
              Final Interview Score + Feedback
```

---

## Models

| Model | Dataset | Architecture | Accuracy |
|---|---|---|---|
| Speech Confidence | RAVDESS (24K samples) | CNN + LSTM (from scratch) | 75%+ |
| Facial Expression | FER2013 (35K images) | Custom CNN (from scratch) | 65%+ |
| Posture Scorer | Self-collected (MediaPipe) | Neural Network (from scratch) | 90%+ |
| Eye Contact | Generated (Face Mesh) | Gaze geometry + classifier | 85%+ |

---

## Tech Stack

Python · TensorFlow · Keras · OpenCV · MediaPipe · Librosa · NumPy · Pandas · Scikit-learn · SHAP · Flask · ReactJS · TailwindCSS · WebRTC

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/AbuBakerAttique/AI-Interview-Coach.git
cd AI-Interview-Coach
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
- RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio → `data/raw/ravdess/`
- FER2013: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset → `data/raw/fer/`

### 4. Collect posture dataset
```bash
python src/data/pose_collector.py
```

### 5. Process datasets
```bash
python src/data/audio_processor.py
python src/data/face_processor.py
```

### 6. Train models
```bash
python src/training/train_speech.py
python src/training/train_face.py
python src/training/train_posture.py
```

### 7. Run the app
```bash
# Backend
python app/backend/app.py

# Frontend (in a new terminal)
cd app/frontend
npm install && npm start
```

---

## Project Structure

```
AI Interview Coach/
├── data/               ← Raw and processed datasets
├── notebooks/          ← EDA and training notebooks
├── src/
│   ├── data/           ← Data processors and collectors
│   ├── models/         ← Model architectures (from scratch)
│   ├── training/       ← Training scripts
│   └── inference/      ← Real-time pipeline and score fusion
├── saved_models/       ← Trained model weights
├── app/
│   ├── backend/        ← Flask API
│   └── frontend/       ← React interface
├── reports/            ← Metrics, curves, logs
└── requirements.txt
```

---

## Author

**Abubaker Attique**
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com/AbuBakerAttique)
