# AI Interview Coach — Full Project Plan

## Project Overview

A real-time, multi-modal AI system that analyzes a user's interview performance through their webcam and microphone. It uses 4 custom-trained deep learning models to evaluate speech confidence, facial expressions, body posture, and eye contact — producing a final interview confidence score with detailed feedback.

**This is not a semester project. This is a production-level AI system.**

---

## What Makes This Impressive for AI Engineer Roles

- 4 separate deep learning models trained from scratch (not APIs, not pre-trained weights)
- Multi-modal AI pipeline (audio + vision + pose combined)
- Real-time inference via webcam and microphone
- Self-collected dataset for posture model (shows data engineering skill)
- Full deployment: Flask API + React frontend
- SHAP explainability on model predictions

---

## System Architecture

```
Webcam + Mic Input
        │
        ├──► Audio Stream ──► MFCC Extraction ──► Speech CNN/LSTM ──► Confidence Score
        │
        ├──► Face Frames ──► Preprocessing ──► Face CNN ──► Expression Score
        │
        ├──► Pose Frames ──► MediaPipe Landmarks ──► Posture NN ──► Posture Score
        │
        └──► Face Mesh ──► Gaze Tracking ──► Eye Contact Score
                                    │
                                    ▼
                          Score Fusion Layer
                                    │
                                    ▼
                      Final Interview Confidence Score
                      + Real-time Feedback + Report PDF
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Audio Processing | Librosa, SoundDevice |
| Computer Vision | OpenCV, MediaPipe |
| Data Science | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| Explainability | SHAP |
| Backend API | Flask / FastAPI |
| Frontend | ReactJS, TailwindCSS, WebRTC |
| Versioning | Git, GitHub |

---

## Models — From Scratch

### Model 1: Speech Confidence Classifier (CNN on MFCC)
- **Goal:** Classify audio as Confident / Nervous / Neutral
- **Architecture:** 1D CNN or LSTM trained on MFCC features
- **Input:** Raw .wav audio → MFCC (40 coefficients) → (Time x 40) matrix
- **Output:** Confidence class + probability score
- **Custom layers:** Conv1D → BatchNorm → MaxPool → LSTM → Dense → Softmax

### Model 2: Facial Expression Classifier (Custom CNN)
- **Goal:** Classify face as Calm / Stressed / Nervous / Confident
- **Architecture:** Custom CNN (NOT VGG, NOT ResNet — built from scratch)
- **Input:** 48x48 grayscale face images
- **Output:** 7-class expression probabilities, mapped to interview states
- **Custom layers:** Conv2D → BatchNorm → MaxPool (x4) → Flatten → Dense → Dropout → Softmax

### Model 3: Posture Scorer (Neural Network on Landmarks)
- **Goal:** Score posture quality (0–100) for interview setting
- **Architecture:** Feedforward Neural Network on 33 pose landmarks (x,y,z = 99 features)
- **Input:** MediaPipe pose landmark array
- **Output:** Posture score (regression) or Good/Bad/Moderate (classification)
- **Custom layers:** Dense(256) → ReLU → Dropout → Dense(128) → ReLU → Dense(1 or 3)

### Model 4: Eye Contact Detector (Rule-based + Lightweight Classifier)
- **Goal:** Detect if candidate is making eye contact with camera
- **Architecture:** MediaPipe Face Mesh → iris landmark ratios → Logistic Regression or small NN
- **Input:** 468 face mesh landmarks, focus on iris (landmarks 468–477)
- **Output:** Eye Contact score (0–100)

---

## Datasets

### Dataset 1: RAVDESS — Speech Emotion Recognition
- **Link:** https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- **Size:** ~24,000 audio samples, 24 actors, 8 emotions
- **Download to:** `data/raw/ravdess/`
- **Labels we use:** calm, happy → Confident | fearful, sad → Nervous | neutral → Neutral
- **Format:** .wav files, 3–5 seconds each

### Dataset 2: FER2013 — Facial Expression Recognition
- **Link:** https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
- **Size:** 35,887 grayscale 48x48 face images
- **Download to:** `data/raw/fer/`
- **Labels we use:** 7 classes → remap to interview-relevant states
- **Format:** Folders per class (angry, disgust, fear, happy, neutral, sad, surprise)

### Dataset 3: Posture — SELF COLLECTED
- **No Kaggle needed.** You generate this yourself using MediaPipe.
- **How:** Record 30-minute videos of yourself and others sitting in good/bad interview posture
- **Script:** `src/data/pose_collector.py` extracts landmarks automatically and saves labeled CSV
- **Labels:** good_posture, slouching, crossed_arms, leaning_back, looking_down
- **Why this impresses:** You built your own dataset. Interviewers love this.

### Dataset 4: Eye Contact — Generated from Face Mesh
- No external dataset. MediaPipe Face Mesh gives iris landmarks in real-time.
- Gaze ratio is computed geometrically (iris position relative to eye corners)

---

## Project Phases & Timeline

### Phase 1: Setup & Data (Week 1)
- [ ] Set up virtual environment and install dependencies
- [ ] Download RAVDESS dataset from Kaggle
- [ ] Download FER2013 dataset from Kaggle
- [ ] Run `pose_collector.py` to record and label your own posture dataset (30 mins)
- [ ] Explore all datasets in notebooks (EDA)

### Phase 2: Audio Model — Speech Confidence (Week 2)
- [ ] `notebooks/01_speech_eda.ipynb` — Explore RAVDESS, visualize waveforms and spectrograms
- [ ] `src/data/audio_processor.py` — Extract MFCC features, augment (noise, pitch shift)
- [ ] `src/models/speech_model.py` — Define CNN/LSTM architecture from scratch
- [ ] `src/training/train_speech.py` — Train model, save best weights
- [ ] Target accuracy: 75%+ on validation set
- [ ] Log metrics to `reports/model_metrics/speech_metrics.json`

### Phase 3: Face Model — Expression Classifier (Week 3)
- [ ] `notebooks/02_face_eda.ipynb` — Explore FER2013, class distribution, sample images
- [ ] `src/data/face_processor.py` — Preprocess images, augmentation (flip, rotate, brightness)
- [ ] `src/models/face_model.py` — Define custom CNN architecture from scratch
- [ ] `src/training/train_face.py` — Train model with class balancing (SMOTE or weighted loss)
- [ ] Target accuracy: 65%+ (FER2013 is a hard dataset, 65% is strong)
- [ ] Log metrics to `reports/model_metrics/face_metrics.json`

### Phase 4: Posture Model (Week 4)
- [ ] `notebooks/03_posture_data_collection.ipynb` — Run collection script, review labels
- [ ] `src/data/pose_collector.py` — Extract landmarks from video, save CSV
- [ ] `src/models/posture_model.py` — Define Neural Network on landmark features
- [ ] `src/training/train_posture.py` — Train and evaluate
- [ ] Target accuracy: 90%+ (clean landmark features, simple classes)
- [ ] Log metrics to `reports/model_metrics/posture_metrics.json`

### Phase 5: Eye Contact & Score Fusion (Week 5)
- [ ] `src/models/eye_contact.py` — Implement iris gaze ratio using Face Mesh
- [ ] `src/inference/score_calculator.py` — Combine all 4 scores into final confidence score
- [ ] Define weights: Speech 30% + Face 25% + Posture 25% + Eye Contact 20%
- [ ] Add SHAP explainability to show what drove the score

### Phase 6: Real-Time System (Week 6)
- [ ] `src/inference/real_time_analyzer.py` — Integrate all models for live webcam + mic
- [ ] `app/backend/app.py` — Flask API to serve predictions
- [ ] `app/frontend/` — React interface with live video feed and real-time scores
- [ ] Generate PDF report after each session

### Phase 7: Polish & Deploy (Week 7)
- [ ] Write comprehensive README with screenshots and GIFs
- [ ] Add model comparison tables in reports folder
- [ ] Docker containerize the backend
- [ ] Deploy backend to AWS or Render
- [ ] Record a demo video for GitHub

---

## Folder Structure

```
AI Interview Coach/
├── data/
│   ├── raw/
│   │   ├── ravdess/               ← Download from Kaggle
│   │   ├── fer/                   ← Download from Kaggle
│   │   └── posture/               ← Your recorded videos
│   ├── processed/
│   │   ├── audio_features/        ← Extracted MFCC .npy files
│   │   ├── face_features/         ← Preprocessed face images
│   │   └── pose_landmarks/        ← Landmark CSV files
│   └── collected/                 ← Self-generated posture labels
│
├── notebooks/
│   ├── 01_speech_eda.ipynb
│   ├── 02_face_eda.ipynb
│   ├── 03_posture_data_collection.ipynb
│   ├── 04_speech_model_training.ipynb
│   ├── 05_face_model_training.ipynb
│   ├── 06_posture_model_training.ipynb
│   └── 07_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── audio_processor.py     ← MFCC extraction + augmentation
│   │   ├── face_processor.py      ← Image preprocessing + augmentation
│   │   └── pose_collector.py      ← Record + label posture dataset
│   ├── models/
│   │   ├── speech_model.py        ← CNN/LSTM architecture (from scratch)
│   │   ├── face_model.py          ← CNN architecture (from scratch)
│   │   ├── posture_model.py       ← Neural Network (from scratch)
│   │   └── eye_contact.py         ← Gaze tracker using Face Mesh
│   ├── training/
│   │   ├── train_speech.py        ← Train speech model
│   │   ├── train_face.py          ← Train face model
│   │   └── train_posture.py       ← Train posture model
│   ├── inference/
│   │   ├── real_time_analyzer.py  ← Live webcam + mic pipeline
│   │   └── score_calculator.py   ← Fuse all scores + SHAP
│   └── utils/
│       ├── audio_utils.py
│       ├── video_utils.py
│       └── metrics.py
│
├── saved_models/
│   ├── speech_confidence/         ← speech_model.h5
│   ├── facial_expression/         ← face_model.h5
│   ├── posture_scorer/            ← posture_model.h5
│   └── eye_contact/               ← gaze_model.pkl
│
├── app/
│   ├── backend/
│   │   ├── app.py                 ← Flask API
│   │   ├── routes/
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/                   ← React components
│       └── public/
│
├── reports/
│   ├── model_metrics/             ← Accuracy, loss, confusion matrices
│   └── training_logs/             ← TensorBoard logs
│
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md                ← This file
```

---

## Model Performance Targets

| Model | Dataset | Target Accuracy | Notes |
|---|---|---|---|
| Speech Confidence | RAVDESS | 75%+ | Hard due to actor variability |
| Facial Expression | FER2013 | 65%+ | FER2013 is notoriously noisy |
| Posture Scorer | Self-collected | 90%+ | Clean landmark features |
| Eye Contact | Generated | 85%+ | Rule-based + classifier |

---

## What to Highlight on Your CV

```
AI Interview Coach — Real-Time Multi-Modal Interview Analysis System        2026
github.com/AbuBakerAttique/AI-Interview-Coach

- Engineered a multi-modal AI pipeline combining 4 custom-trained deep learning
  models (Speech CNN/LSTM, Facial Expression CNN, Posture NN, Gaze Classifier)
  for real-time interview performance analysis.
- Trained Speech Confidence model from scratch on RAVDESS dataset (24K samples)
  using MFCC feature extraction achieving 75%+ accuracy.
- Trained Facial Expression CNN from scratch on FER2013 (35K+ images) for
  real-time stress and confidence detection.
- Collected and labeled custom posture dataset using MediaPipe Pose,
  training a Neural Network achieving 90%+ accuracy.
- Built score fusion layer with SHAP explainability, combining all model
  outputs into a final confidence score with actionable feedback.
- Deployed full-stack system: Flask API backend + React frontend with WebRTC
  real-time video/audio streaming.
```

---

## Dependencies (requirements.txt)

```
tensorflow>=2.13.0
torch>=2.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
librosa>=0.10.0
sounddevice>=0.4.6
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
flask>=2.3.0
flask-cors>=4.0.0
imbalanced-learn>=0.11.0
jupyter>=1.0.0
tqdm>=4.65.0
```

---

## First Steps — Start Today

1. Download RAVDESS from Kaggle → place in `data/raw/ravdess/`
2. Download FER2013 from Kaggle → place in `data/raw/fer/`
3. Run: `pip install -r requirements.txt`
4. Open `notebooks/01_speech_eda.ipynb` and start exploring the audio data
5. Begin coding `src/data/audio_processor.py`

---

*Built by Abubaker Attique — 2026*
