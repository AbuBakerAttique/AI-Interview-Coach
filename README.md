# AI Interview Coach

A real-time, multi-modal AI system that analyzes interview performance through webcam and microphone. Four custom-trained deep learning models evaluate speech confidence, facial expressions, body posture, and eye contact — producing a final confidence score with actionable feedback.

---

## Demo

> *[Add GIF or screenshot of the live system here]*

---

## Model Results

| Model | Dataset | Architecture | Accuracy |
|---|---|---|---|
| Speech Confidence | RAVDESS (2,880 audio files) | CNN + LSTM (from scratch) | **93.40%** |
| Facial Expression | FER2013 (28,821 images) | Custom CNN (from scratch) | **75.54%** |
| Posture Scorer | MultiPosture (4,794 samples) | Neural Network (from scratch) | **97.29%** |
| Eye Contact | MediaPipe Face Mesh | Iris gaze geometry | Rule-based |

> Note: 75.54% on FER2013 is a strong result — it is one of the most challenging face datasets in the field. Top published research achieves 65–75% on the same dataset.

---

## How It Works

```
Webcam + Mic
    │
    ├── Audio → MFCC → Speech CNN/LSTM     → Confidence Score (30%)
    ├── Face  → CNN  → Expression Score    (25%)
    ├── Pose  → MediaPipe → Posture NN     → Posture Score    (25%)
    └── Iris  → Face Mesh → Eye Contact    → Eye Score        (20%)
                          │
                   Score Fusion Layer
                          │
              Final Interview Score (0-100) + Grade + Feedback
```

---

## Tech Stack

Python · PyTorch · OpenCV · MediaPipe · Librosa · NumPy · Pandas · Scikit-learn · Flask · ReactJS · TailwindCSS

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/AbuBakerAttique/AI-Interview-Coach.git
cd AI-Interview-Coach
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets

| Dataset | Link | Place in |
|---|---|---|
| RAVDESS (Speech) | [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | `data/raw/ravdess/` |
| FER2013 (Face) | [Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) | `data/raw/fer/` |
| MultiPosture (Posture) | [Zenodo](https://zenodo.org/records/14230872) | `data/raw/posture/` |

### 5. Train the models

Run each training script in order:

```bash
# Train posture model (~2 mins)
python src/training/train_posture.py

# Train speech confidence model (~10 mins)
python src/training/train_speech.py

# Train facial expression model (~15 mins)
python src/training/train_face.py
```

Expected results after training:

```
Posture Model     → ~97% validation accuracy
Speech Model      → ~93% validation accuracy
Face Model        → ~75% validation accuracy (strong result for FER2013)
```

### 6. Run the app
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
├── data/
│   ├── raw/
│   │   ├── ravdess/        ← RAVDESS audio dataset
│   │   ├── fer/            ← FER2013 face images
│   │   └── posture/        ← MultiPosture landmark data
│   └── processed/          ← Processed features
│
├── notebooks/
│   ├── 01_speech_eda.ipynb ← Explore audio data
│   └── 02_face_eda.ipynb   ← Explore face images
│
├── src/
│   ├── data/               ← Data processors
│   ├── models/             ← Model architectures (built from scratch)
│   ├── training/           ← Training scripts
│   └── inference/          ← Real-time pipeline and score fusion
│
├── saved_models/           ← Trained model weights (not on GitHub)
├── app/
│   ├── backend/            ← Flask API
│   └── frontend/           ← React interface
├── reports/                ← Training curves and metrics
└── requirements.txt
```

---

## What I Learned Building This

- How to process raw audio into MFCC features using Librosa
- How to build CNN + LSTM architectures from scratch in PyTorch
- How to handle imbalanced datasets using class weights
- How to use MediaPipe for real-time pose and face mesh detection
- How to fuse multiple model outputs into a single score
- How to deploy AI models via Flask API

---

## Author

**Abubaker Attique**
Computer Science Graduate — NUCES FAST, Islamabad
[LinkedIn](https://linkedin.com) | [GitHub](https://github.com/AbuBakerAttique)
