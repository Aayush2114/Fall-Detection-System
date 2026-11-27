# 🏥 AI-Based Human Fall Detection System Using Computer Vision and Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-0097A7.svg)](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: CC BY 4.0 (Dataset)](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Educational Research Prototype](https://img.shields.io/badge/Status-Educational%20Research%20Prototype-orange.svg)](#limitations)

> ⚠️ **Important Disclaimer:** This project is an **educational research prototype** built for academic learning and experimentation. It is **NOT** a medically validated safety system, clinical diagnostic tool, or life-safety alerting product, and must not be relied upon as a primary or sole emergency monitoring solution.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Why This Project Is Useful](#why-this-project-is-useful)
4. [Project Objectives](#project-objectives)
5. [Technologies Used](#technologies-used)
6. [System Architecture](#system-architecture)
7. [Complete Workflow](#complete-workflow)
8. [Dataset Details](#dataset-details)
9. [Installation Instructions](#installation-instructions)
10. [How to Run the Project](#how-to-run-the-project)
11. [How to Train the Model](#how-to-train-the-model)
12. [How to Evaluate the Model](#how-to-evaluate-the-model)
13. [How to Run Inference](#how-to-run-inference)
14. [Expected Output Examples (Illustrative Only)](#expected-output-examples-illustrative-only)
15. [Evaluation Metrics](#evaluation-metrics)
16. [Limitations](#limitations)
17. [Future Improvements](#future-improvements)
18. [Project Structure](#project-structure)
19. [References and Citations](#references-and-citations)

---

## 📖 Project Overview

The **AI-Based Human Fall Detection System** is an end-to-end computer vision and deep learning pipeline designed to detect human falls from video streams and pose sequences. Rather than feeding raw RGB pixels directly into heavy 3D convolutional networks, the system extracts compact 3D human skeletal landmarks per frame using **MediaPipe Pose** (33 body joints × 4 attributes = 132 features per frame) and models how body posture evolves over time using a stacked **Long Short-Term Memory (LSTM)** neural network implemented in **PyTorch**.

Because a single static frame cannot reliably distinguish between a person bending down, lying down intentionally, or collapsing during a fall, the system groups consecutive frames into overlapping **30-frame sliding windows** (approximately 1 second of motion at 30 FPS, with a stride of 15 frames). Each sequence is normalized using a `MinMaxScaler` fitted strictly on training data and passed through the LSTM classifier to produce probabilities for two classes: **`Normal Activity`** (`0`) and **`Potential Fall`** (`1`).

The repository is structured for clarity, reproducibility, and hands-on experimentation. It supports subject-wise data splitting on the **UP-Fall Detection Dataset (3D Skeletons variant)** to prevent data leakage, includes a **synthetic data generator** (`--synthetic`) for verifying the pipeline without downloading large datasets, provides detailed evaluation plots (confusion matrix, loss curves, accuracy curves), and offers both CLI inference (pre-recorded video and live webcam) and an interactive terminal demo (`app/demo.py`).

---

## ❗ Problem Statement

Falls are one of the leading causes of accidental injury, loss of independence, and injury-related mortality among older adults worldwide:

- **Elderly Care & Independent Living:** As global life expectancy rises, a growing number of older adults live alone or spend significant periods unsupervised. When a fall occurs and the individual is unconscious, immobilized, or unable to reach a phone, delayed discovery turns a manageable incident into a medical emergency.
- **Response Time & Long-Lie Complications:** Prolonged time spent on the floor after a fall ("long-lie") is strongly associated with severe secondary complications, including hypothermia, dehydration, rhabdomyolysis, pressure ulcers, and psychological trauma (fear of falling again). Rapid automated detection can significantly reduce response latency.
- **Healthcare Costs & Caregiver Burden:** Emergency hospitalizations, hip fractures, prolonged rehabilitation, and round-the-clock manual supervision place substantial financial and operational pressure on families, hospitals, and assisted-living facilities.
- **Limitations of Existing Solutions:** Wearable panic buttons and accelerometer pendants require the user to remember to wear, charge, and—in the case of manual buttons—press the device after a fall. Vision-based pose estimation offers a **non-intrusive, contactless** alternative that requires no body-worn hardware and preserves greater privacy when processing skeletal coordinates rather than storing raw video.

---

## 💡 Why This Project Is Useful

By combining lightweight pose estimation with temporal sequence modeling, this prototype demonstrates practical building blocks for several real-world domains (subject to further clinical validation and engineering hardening):

- **🏠 Smart Home Elderly Monitoring:** Enables contactless monitoring in living rooms or hallways using standard off-the-shelf cameras without forcing residents to wear sensors.
- **🏥 Assisted Living & Healthcare Facilities:** Demonstrates how automated video analytics could assist nursing staff in prioritizing room checks when potential falls occur across multiple care spaces.
- **🔒 Privacy-Preserving Analytics:** Converting video frames into 33 skeletal coordinates discards sensitive visual details (facial identity, clothing, room contents) before downstream classification.
- **🎓 Academic & Research Benchmark:** Provides a clean, modular PyTorch + MediaPipe codebase for studying temporal human activity recognition (HAR), subject-independent evaluation, and class-imbalance metrics.

---

## 🎯 Project Objectives

- **Pose Landmark Extraction:** Extract 33 body landmarks (`x`, `y`, `z`, `visibility`) per frame from video files and live webcam feeds using OpenCV and MediaPipe Pose.
- **Leakage-Free Preprocessing:** Normalize 132-dimensional frame vectors with `MinMaxScaler` fitted exclusively on training subjects, and segment continuous recordings into overlapping 30-frame temporal windows (`stride=15`).
- **Temporal Deep Learning Model:** Train a 2-layer stacked PyTorch LSTM classifier (`hidden_size=128`, `dropout=0.3`, fully connected head `128 → 32 → 2`) with Adam optimization, cross-entropy loss, and early stopping.
- **Subject-Independent Evaluation:** Split the UP-Fall dataset by Subject ID (Subjects 1–11 for training, 12–14 for validation, 15–17 for testing) so the model is evaluated on unseen individuals.
- **Comprehensive Metrics & Diagnostics:** Compute Accuracy, Precision, Recall, and F1-Score, and export confusion matrices and training/validation curves—emphasizing **Recall** to minimize missed falls.
- **End-to-End Inference & Usability:** Provide real-time webcam inference, annotated video file inference with visual fall alert banners, a synthetic-data testing mode, and an interactive CLI demo application.

---

## 🛠️ Technologies Used

Version requirements below match [`requirements.txt`](requirements.txt) in this repository.

| Technology | Version | Purpose & Why It Is Used |
| :--- | :--- | :--- |
| **Python** | `3.9+` (Python 3) | Core programming language; rich ecosystem for computer vision, deep learning, and scientific computing. |
| **PyTorch** (`torch`, `torchvision`, `torchaudio`) | `>= 2.0.0` (`torchvision >= 0.15.0`) | Deep learning framework used to build, train, save, and run the `FallDetectionLSTM` sequence classifier with optional CUDA GPU acceleration. |
| **OpenCV** (`opencv-python`) | `>= 4.8.0` | Video decoding, live webcam capture, BGR/RGB conversion, bounding/alert overlays, and writing annotated `.mp4` output videos. |
| **MediaPipe** (`mediapipe`) | `>= 0.10.0` | Real-time human pose estimation (BlazePose) extracting 33 3D skeletal landmarks (`x, y, z, visibility`) per frame on CPU or GPU. |
| **NumPy** | `>= 1.24.0` | Fast array operations for 132-feature frame vectors, sliding-window sequence construction, linear interpolation, and synthetic data generation. |
| **Pandas** | `>= 2.0.0` | Reading and parsing CSV landmark files from the UP-Fall 3D Skeletons dataset. |
| **scikit-learn** | `>= 1.3.0` | `MinMaxScaler` for `[0, 1]` feature normalization and metric computation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`). |
| **Matplotlib** | `>= 3.7.0` | Generating and saving non-interactive evaluation plots (`confusion_matrix.png`, `loss_curves.png`, `accuracy_curves.png`). |
| **tqdm** | `>= 4.65.0` | Progress bars for long-running loops and data workflows. |
| **Jupyter / IPyKernel** | `>= 1.0.0` / `>= 6.0.0` | Optional interactive exploration and experimentation in `notebooks/`. |

---

## 🏗️ System Architecture

The end-to-end processing pipeline transforms raw visual frames into temporal fall predictions:

```mermaid
flowchart LR
    A["🎥 Video / Webcam Input"] --> B["📷 OpenCV (Frame Capture & BGR to RGB)"]
    B --> C["🦴 MediaPipe Pose (33 Body Landmarks)"]
    C --> D["📐 Feature Extraction (33 x 4 = 132 Features)"]
    D --> E["⚖️ MinMaxScaler (Normalize to 0..1)"]
    E --> F["🪟 Sliding Window (30 Frames, Stride 15)"]
    F --> G["🧠 2-Layer LSTM + FC (128 -> 32 -> 2)"]
    G --> H["📊 Softmax Prediction (Normal vs Potential Fall)"]
    H --> I["🖥️ Annotated Output (Overlay, Alert & Stats)"]
```

### Model Architecture Summary (`src/model.py`)

```text
Input Tensor: (batch_size, 30, 132)
   │
   ▼
nn.LSTM(input_size=132, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
   │
   ▼  Take final time step: lstm_out[:, -1, :] → (batch_size, 128)
nn.Dropout(p=0.3)
   │
   ▼
nn.Linear(128 → 32) + nn.ReLU()
   │
   ▼
nn.Linear(32 → 2)  → Logits [Normal Activity, Potential Fall]
   │
   ▼  (At inference / evaluation)
torch.softmax(logits, dim=1) → Class probabilities & confidence threshold check
```

---

## 🔄 Complete Workflow

1. **Video Capture (`src/pose_extraction.py` / `src/inference.py`):** OpenCV (`cv2.VideoCapture`) reads frames sequentially from a `.mp4`/`.avi` video file or a connected webcam (`index 0`). Each frame is converted from BGR to RGB.
2. **Pose Estimation (`MediaPipe Pose`):** Each RGB frame is processed by `mp.solutions.pose.Pose` (`model_complexity=1`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`). MediaPipe detects **33 anatomical landmarks** (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.).
3. **Feature Vector Construction:** For each detected frame, all 33 landmarks are flattened into a `132`-element `float32` vector containing `[x, y, z, visibility]` per joint. During offline video extraction, `interpolate_missing_frames()` in `src/preprocessing.py` can linearly fill frames where pose detection temporarily drops.
4. **Data Splitting & Leakage Prevention (`src/dataset.py`):** When using the UP-Fall dataset, recordings are partitioned strictly **by Subject ID** (`1–11` Train, `12–14` Validation, `15–17` Test) rather than randomly shuffling frames/windows, ensuring the model generalizes to unseen people.
5. **Feature Normalization (`src/preprocessing.py`):** A `MinMaxScaler` scales all 132 features into `[0, 1]`. Crucially, the scaler is **fitted on training frames only** (`fit=True`) and reused via `transform()` on validation, test, and live inference data.
6. **Temporal Sequence Windowing (`src/preprocessing.py`):** Frame arrays are segmented into fixed-length sequences of `30` consecutive frames (`SEQUENCE_LENGTH = 30`, ~1 second at 30 FPS) with a `stride` of `15` frames (`SEQUENCE_STRIDE = 15`, 50% overlap) during dataset preparation, or via a rolling `deque(maxlen=30)` buffer during real-time inference.
7. **LSTM Classification (`src/model.py` & `src/train.py`):** Sequences of shape `(batch, 30, 132)` pass through the 2-layer stacked LSTM. Training uses `CrossEntropyLoss`, the `Adam` optimizer (`lr=0.001`), up to `50` epochs (`batch_size=32`), and early stopping (`patience=10`) while saving the checkpoint with the lowest validation loss (`models/fall_detection_lstm.pth`).
8. **Inference & Alert Visualization (`src/inference.py`):** At runtime, once the 30-frame buffer is full, `model.predict_proba()` computes probabilities for `Normal Activity` and `Potential Fall`. The skeleton, predicted status, and confidence percentage are drawn onto each frame. If `Potential Fall` exceeds `--threshold` (default `0.6`), a red **`FALL DETECTED — ALERT`** banner is overlaid and the annotated video is saved to `outputs/`.

---

## 📊 Dataset Details

This project is designed around the **UP-Fall Detection Dataset** created by researchers at **Universidad Panamericana, Mexico**, specifically the **3D Skeletons variant** hosted on Zenodo, which provides MediaPipe BlazePose 3D landmarks directly compatible with this pipeline.

### Dataset Summary

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | UP-Fall Detection Dataset (3D Skeletons Variant) |
| **Subjects** | **17 healthy young subjects** (aged 18–24) |
| **Activities** | **11 activities** total: **5 fall types** (Activities 1–5) + **6 daily living activities** (Activities 6–11) |
| **Trials** | **3 trials** per activity per subject |
| **Features** | At least **132 columns** per frame (`33` MediaPipe landmarks × `4` values: `x, y, z, visibility`) |
| **License** | **CC BY 4.0** (3D Skeletons variant on Zenodo — free to share and adapt with attribution); full UP-Fall multimodal dataset is available for academic/research use |
| **Zenodo Link (3D Skeletons)** | [https://zenodo.org/records/12781431](https://zenodo.org/records/12781431) |
| **Original Project Page** | [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/) |

### Activity Label Mapping (`src/config.py`)

| Activity ID | Category | Binary Label | Description |
| :---: | :--- | :---: | :--- |
| `1` | Fall | `1` (`Potential Fall`) | Falling forward using hands |
| `2` | Fall | `1` (`Potential Fall`) | Falling forward using knees |
| `3` | Fall | `1` (`Potential Fall`) | Falling backwards |
| `4` | Fall | `1` (`Potential Fall`) | Falling sideward |
| `5` | Fall | `1` (`Potential Fall`) | Falling sitting in empty chair |
| `6` | Normal | `0` (`Normal Activity`) | Walking |
| `7` | Normal | `0` (`Normal Activity`) | Standing |
| `8` | Normal | `0` (`Normal Activity`) | Sitting |
| `9` | Normal | `0` (`Normal Activity`) | Picking up an object |
| `10` | Normal | `0` (`Normal Activity`) | Jumping |
| `11` | Normal | `0` (`Normal Activity`) | Laying |

### Subject-Wise Split Strategy

- **Training Set:** Subjects `1–11` (~65%)
- **Validation Set:** Subjects `12–14` (~17%)
- **Test Set:** Subjects `15–17` (~18%)

### Download & Setup Instructions

1. **Option A — 3D Skeletons CSVs (Recommended):**
   - Download the 3D Skeletons dataset from Zenodo: [https://zenodo.org/records/12781431](https://zenodo.org/records/12781431)
   - Extract the files into `data/raw/` organized by subject and activity folders so `load_upfall_skeleton_data()` can parse `Subject*` and `Activity*` IDs from the paths:

   ```text
   data/raw/
   ├── Subject1/
   │   ├── Activity1/
   │   │   ├── Trial1.csv
   │   │   ├── Trial2.csv
   │   │   └── Trial3.csv
   │   ├── Activity2/
   │   │   └── ...
   │   └── Activity11/
   ├── Subject2/
   │   └── ...
   └── Subject17/
   ```

2. **Option B — Full Raw Video Dataset (850+ GB, Optional):**
   - Request academic access at [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/) and use `src/pose_extraction.py` to extract 132-feature sequences from videos.

> 💡 **No dataset downloaded yet?** You can run the entire training and evaluation workflow right away using `--synthetic` mode (see [How to Train the Model](#how-to-train-the-model)). See [`data/README.md`](data/README.md) for additional dataset notes.

---

## ⚙️ Installation Instructions

### Prerequisites

- **Python 3.9 or higher** (3.10 / 3.11 recommended for broad MediaPipe and PyTorch wheel compatibility)
- **pip** package manager
- *(Optional)* NVIDIA GPU with compatible CUDA drivers for faster training
- *(Optional)* Webcam for live inference

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fall-detection-system
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade `pip` and Install PyTorch (with CUDA or CPU)

Install PyTorch **before** `requirements.txt` if you want a specific CUDA wheel. Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to match your CUDA toolkit version:

**NVIDIA GPU (example for CUDA 12.1):**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (if no NVIDIA GPU is available):**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

### Step 4: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch, cv2, mediapipe, sklearn; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| OpenCV:', cv2.__version__, '| MediaPipe:', mediapipe.__version__)"
```

---

## 🚀 How to Run the Project

All commands should be executed from the `fall-detection-system/` root directory with your virtual environment activated.

### Quick Command Reference

| Task | Command |
| :--- | :--- |
| **Launch Interactive Demo Menu** | `python app/demo.py` |
| **Train with Synthetic Data (Pipeline Test)** | `python -m src.train --synthetic` |
| **Train with Real UP-Fall Data** | `python -m src.train` |
| **Evaluate on Synthetic Data** | `python -m src.evaluate --synthetic` |
| **Evaluate on Real Test Subjects** | `python -m src.evaluate` |
| **Run Inference on a Video File** | `python -m src.inference --video path/to/video.mp4` |
| **Run Inference on Live Webcam** | `python -m src.inference` |
| **Test Pose Extraction Standalone** | `python -m src.pose_extraction path/to/video.mp4` |

### Interactive Demo App

For a menu-driven experience covering video inference, webcam detection, training, evaluation, and system diagnostics:

```bash
python app/demo.py
```

---

## 🏋️ How to Train the Model

### 1. Pipeline Verification with Synthetic Data (`--synthetic`)

If you have not downloaded the UP-Fall dataset yet, use the `--synthetic` flag:

```bash
python -m src.train --synthetic
```

**What `--synthetic` does:**
- Calls `generate_synthetic_data()` in `src/dataset.py`, creating 200 normal and 200 fall sequences (`(400, 30, 132)`), split `70% / 15% / 15%` into train/validation/test sets.
- Normal sequences simulate gentle sinusoidal oscillations around a base pose, whereas fall sequences simulate a sudden downward shift in MediaPipe `y`-coordinates (`index 1::4`) and depth shift (`index 2::4`).
- **Purpose:** Verifies that PyTorch, data loaders, training loops, early stopping, and checkpoint saving (`models/fall_detection_lstm.pth`) work end-to-end.
- ⚠️ **Warning:** Synthetic accuracy and loss **do NOT represent real-world fall detection capability**. Never report synthetic metrics as real performance.

### 2. Training with the Real UP-Fall Dataset

Once CSV files are placed in `data/raw/` (following the `Subject*/Activity*/Trial*.csv` structure):

```bash
python -m src.train
```

### Optional Training Arguments (`src/train.py`)

| Flag | Default (`src/config.py`) | Description |
| :--- | :---: | :--- |
| `--synthetic` | `False` | Use synthetic data instead of `data/raw/` |
| `--epochs` | `50` | Maximum number of training epochs |
| `--lr` | `0.001` | Learning rate for the Adam optimizer |
| `--batch-size` | `32` | Number of 30-frame sequences per batch |

**Example with custom hyperparameters:**
```bash
python -m src.train --epochs 40 --lr 0.0005 --batch-size 64
```

**Artifacts produced by training:**
- `models/fall_detection_lstm.pth` — Best model weights (lowest validation loss) plus architecture metadata.
- `outputs/training_history.json` — Per-epoch training/validation loss and accuracy logs.
- `data/processed/scaler.pkl` — Fitted `MinMaxScaler` (when training on real UP-Fall data).

> 📝 **Note on Scaler Path for Inference:** `src/train.py` (via `src/dataset.py`) saves the real-data `MinMaxScaler` to `data/processed/scaler.pkl`, while `src/inference.py` defaults `--scaler` to `models/feature_scaler.pkl` (`SCALER_SAVE_PATH`). When running `src.inference` with a real-data-trained model, pass `--scaler data/processed/scaler.pkl` (or copy `data/processed/scaler.pkl` to `models/feature_scaler.pkl`). Note also that `--synthetic` mode generates pre-bounded `[0, 1]` sequences directly without creating a `.pkl` scaler file.

---

## 📈 How to Evaluate the Model

After training, evaluate the saved checkpoint (`models/fall_detection_lstm.pth`) on the held-out test split:

**Evaluate on Synthetic Test Split:**
```bash
python -m src.evaluate --synthetic
```

**Evaluate on Real Test Subjects (Subjects 15–17):**
```bash
python -m src.evaluate
```

**Evaluate a Specific Model Checkpoint:**
```bash
python -m src.evaluate --model models/fall_detection_lstm.pth
```

### Generated Metrics & Artifacts

Running `src/evaluate.py` computes and saves:
1. **Console Report:** Accuracy, Precision, Recall, F1-Score, and `scikit-learn` per-class `classification_report`.
2. **`outputs/evaluation_metrics.json`:** Machine-readable dictionary of `accuracy`, `precision`, `recall`, and `f1_score`.
3. **`outputs/confusion_matrix.png`:** 2×2 Confusion Matrix heatmap (`Normal Activity` vs `Potential Fall`).
4. **`outputs/loss_curves.png` & `outputs/accuracy_curves.png`:** Training vs. validation curves generated from `outputs/training_history.json` (when present).

---

## 🎥 How to Run Inference

Inference requires both a trained model (`.pth`) and a fitted `MinMaxScaler` (`.pkl`).

### Mode 1: Video File Inference

```bash
python -m src.inference --video path/to/video.mp4
```

If your scaler was saved to `data/processed/scaler.pkl` during real-data training:

```bash
python -m src.inference --video path/to/video.mp4 --scaler data/processed/scaler.pkl
```

### Mode 2: Live Webcam Inference

```bash
python -m src.inference
```

**Webcam Keyboard Controls:**
- Press **`q`** to stop inference and close the window.
- Press **`s`** to save a snapshot to `outputs/webcam_screenshot.png`.

### Inference Command-Line Arguments (`src/inference.py`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--video` | `None` | Path to an input video file. Omit to use the default webcam (`cv2.VideoCapture(0)`). |
| `--model` | `models/fall_detection_lstm.pth` | Path to trained PyTorch `.pth` checkpoint. |
| `--scaler` | `models/feature_scaler.pkl` | Path to fitted `MinMaxScaler` `.pkl` file. |
| `--output` | `outputs/output_<video_name>.mp4` | Output path for the annotated video (video mode). |
| `--no-display` | `False` | Disable the live OpenCV preview window (useful for headless/server runs). |
| `--threshold` | `0.6` | Minimum predicted probability required to display the **`FALL DETECTED — ALERT`** banner. |

---

## 🖥️ Expected Output Examples (Illustrative Only)

> ⚠️ **ILLUSTRATIVE EXAMPLES ONLY:** The console logs below are **mocked/illustrative examples** showing the formatting and structure printed by the scripts. They are **not** measured benchmark results from a trained model run.

### Illustrative Training Output (`python -m src.train --synthetic`)

```text
================= Fall Detection Model Training =================
[INFO] Random seed set to 42
[INFO] CUDA not available. Using CPU.

======================= Data Preparation ========================
[INFO] Using SYNTHETIC data (for pipeline testing only)
[WARNING] Results with synthetic data do NOT represent real performance!

[INFO] Generated synthetic data:
  Normal: 200, Fall: 200, Total: 400
  Shape: sequences=(400, 30, 132), labels=(400,)

[INFO] Data splits:
  Train: 280 sequences
  Val:   60 sequences
  Test:  60 sequences
...
=========================== Training ============================
Epochs: 50, LR: 0.001, Batch: 32
Early stopping patience: 10

Epoch   1/50 | Train Loss: 0.68xx, Acc: 0.5xxx | Val Loss: 0.64xx, Acc: 0.6xxx ← Best model saved!
Epoch   2/50 | Train Loss: 0.52xx, Acc: 0.7xxx | Val Loss: 0.41xx, Acc: 0.8xxx ← Best model saved!
...
[INFO] Model saved: models/fall_detection_lstm.pth
[INFO] Saved: outputs/training_history.json
```

### Illustrative Evaluation Output (`python -m src.evaluate --synthetic`)

```text
======================= Model Evaluation ========================
[INFO] Model loaded from models/fall_detection_lstm.pth
[INFO] Architecture: input=132, hidden=128, layers=2, classes=2
[INFO] Running predictions on test data...

====================== Evaluation Results =======================
  Accuracy:  0.xxxx (xx.x%)
  Precision: 0.xxxx
  Recall:    0.xxxx
  F1-Score:  0.xxxx

  Classification Report:
                 precision    recall  f1-score   support

Normal Activity       x.xx      x.xx      x.xx        xx
 Potential Fall       x.xx      x.xx      x.xx        xx

       accuracy                           x.xx        xx
      macro avg       x.xx      x.xx      x.xx        xx
   weighted avg       x.xx      x.xx      x.xx        xx

  ⚠️  Note on Recall:
  In fall detection, RECALL is the most critical metric.
  A False Negative (missed fall) means a person who fell
  does not receive help. A False Positive (false alarm) is
  inconvenient but not dangerous.
  Aim for high recall, even at the cost of some precision.

[INFO] Confusion matrix saved: outputs/confusion_matrix.png
[INFO] Loss plot saved: outputs/loss_curves.png
[INFO] Accuracy plot saved: outputs/accuracy_curves.png
[INFO] Saved: outputs/evaluation_metrics.json
```

### Illustrative Video Inference Output (`python -m src.inference --video sample.mp4`)

```text
[INFO] Model loaded from models/fall_detection_lstm.pth
[INFO] Scaler loaded: models/feature_scaler.pkl
[INFO] Input: sample.mp4 (1280×720 @ 30 FPS)
  Progress: 200/450 (44%)
  Progress: 400/450 (89%)

[INFO] Inference complete!
  Output saved: outputs/output_sample.mp4
  Frames processed: 450
  Frames with pose: 442
  Predictions made: 413
  Fall detections: xx
  Normal detections: xxx
```

---

## 📏 Evaluation Metrics

Understanding how the model is measured is essential in safety-sensitive applications:

| Metric | Formula | What It Measures |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction of sequences classified correctly. Can look artificially high if most sequences are `Normal Activity` (class imbalance). |
| **Precision** | $\frac{TP}{TP + FP}$ | Of all sequences flagged as `Potential Fall`, how many were real falls. High precision means fewer false alarms. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Of all **actual falls**, what fraction did the system successfully detect. High recall means fewer missed falls. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall, balancing missed falls against false alarms. |

### Confusion Matrix Breakdown

| | Predicted: `Normal Activity` (`0`) | Predicted: `Potential Fall` (`1`) |
| :--- | :--- | :--- |
| **Actual: `Normal Activity` (`0`)** | **True Negative (TN)** — Correctly ignored normal motion | **False Positive (FP)** — False alarm |
| **Actual: `Potential Fall` (`1`)** | **False Negative (FN)** — **Missed fall (highest risk)** | **True Positive (TP)** — Correctly detected fall |

### Why Recall Matters Most in Fall Detection

In fall detection, errors are **asymmetric**:
- A **False Positive (FP)** triggers an unnecessary alert when someone simply sat down quickly or bent over—inconvenient, yet harmless and easily dismissed.
- A **False Negative (FN)** fails to alert when an actual fall occurs, potentially leaving an injured person on the floor without timely assistance.

Therefore, **Recall** is the primary safety metric to optimize, tuning the decision threshold (`--threshold`) and class weights so real falls are rarely missed while keeping false alarms manageable.

---

## ⚠️ Limitations

This project is an **educational research prototype**, and its current scope has important technical and practical limitations:

1. **Not Medically Validated:** This software has **not** undergone clinical trials, regulatory review (e.g., FDA/CE medical device certification), or fault-tolerant reliability engineering. It must **never** be deployed as a medical-grade or sole emergency alert system.
2. **Synthetic Data Limitations:** The `--synthetic` mode generates simplified mathematical approximations (downward `y`-coordinate shifts) solely to verify code execution. Synthetic performance does not reflect real human biomechanics or camera noise.
3. **Single-Person Detection Only:** `mp.solutions.pose.Pose` tracks **one prominent person** per frame. In multi-person rooms (caregivers, visitors, pets), pose tracking may jump between individuals or miss the person who fell.
4. **Controlled Dataset Demographics & Environments:** The UP-Fall dataset consists of **17 young, healthy volunteers** performing staged falls onto protective mats under controlled lab lighting and fixed camera viewpoints. Performance may degrade significantly on elderly individuals, varied body types, loose clothing, low-light rooms, or heavy occlusion (furniture, blankets).
5. **Viewpoint & Frame-Rate Sensitivity:** Raw `x, y, z` coordinates depend on camera angle, distance, and frame rate (`SEQUENCE_LENGTH=30` assumes ~30 FPS). Rapidly lying down or sitting down can occasionally resemble falling without engineered velocity/posture-recovery features.
6. **Inference Scaler Alignment:** During real-data training, `src/dataset.py` saves the scaler to `data/processed/scaler.pkl`, whereas `src/inference.py` looks at `models/feature_scaler.pkl` by default unless `--scaler` is explicitly specified.

---

## 🔮 Future Improvements

- **Multi-Person Tracking:** Integrate a person detector and tracker (e.g., YOLO + ByteTrack/DeepSORT) before pose estimation so each individual in a frame has an independent 30-frame skeleton buffer.
- **View-Invariant Feature Engineering:** Compute hip-centered relative coordinates, torso angle relative to vertical, bounding-box aspect ratio, and inter-frame joint velocity/acceleration instead of raw camera-frame coordinates.
- **Architecture Comparisons:** Benchmark Bidirectional LSTMs, GRUs, 1D Temporal CNNs (TCNs), and Lightweight Pose Transformers / Spatio-Temporal Graph Convolutional Networks (ST-GCN) against the baseline 2-layer LSTM.
- **Post-Fall Inactivity Verification:** Combine fall transition detection with a post-impact stillness check (e.g., lying horizontally for $N$ seconds) to dramatically reduce false positives from fast sitting or lying down.
- **Edge & Embedded Deployment:** Export trained models to ONNX / TorchScript / TensorRT or TFLite for low-power edge devices (Raspberry Pi, NVIDIA Jetson Nano/Orin) with privacy-first local processing.
- **Real-World & Cross-Dataset Testing:** Validate across multiple datasets (e.g., UR Fall, Le2i, NTU RGB+D) and diverse camera angles, lighting conditions, and occluded home environments.
- **Alerting & Integrations:** Add cooldown timers, webhook/MQTT notifications, and caregiver dashboard integration for prototype smart-home testing.

---

## 📁 Project Structure

```text
fall-detection-system/
├── README.md                          # Main project documentation (this file)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignores venv, raw/processed data, models, outputs
├── data/
│   ├── README.md                      # Dataset download, structure, and licensing guide
│   ├── raw/                           # Place UP-Fall 3D Skeletons CSVs here
│   └── processed/                     # Generated scaler / processed artifacts
├── notebooks/                         # Jupyter notebooks for exploration & experiments
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized paths, hyperparameters, and labels
│   ├── pose_extraction.py             # MediaPipe Pose extraction from videos & webcam
│   ├── preprocessing.py               # MinMax normalization, sliding windows, interpolation
│   ├── dataset.py                     # PyTorch Dataset, UP-Fall loader, synthetic generator
│   ├── model.py                       # FallDetectionLSTM architecture + save/load helpers
│   ├── train.py                       # Training loop, validation, early stopping, CLI
│   ├── evaluate.py                    # Metrics calculation, classification report, plots
│   ├── inference.py                   # Video & webcam inference with alert overlays
│   └── utils.py                       # Random seed, GPU/CPU device selection, JSON/timer utils
├── app/
│   └── demo.py                        # Interactive CLI demonstration menu
├── models/                            # Saved model checkpoints (.pth) & scaler (.pkl)
├── outputs/                           # Saved plots (.png), JSON reports, annotated videos
└── docs/
    ├── PROJECT_EXPLANATION.md         # Detailed project walkthrough & rationale
    ├── TECHNICAL_CONCEPTS.md          # Deep dive into MediaPipe, LSTMs, and sequence modeling
    ├── INSTALLATION_GUIDE.md          # Platform-specific setup & troubleshooting guide
    ├── RESULTS_AND_LIMITATIONS.md     # Evaluation methodology, failure modes & analysis
    └── INTERVIEW_PREPARATION.md       # Key technical Q&A and architecture talking points
```

---

## 📚 References and Citations

1. **UP-Fall Detection Dataset (Original Paper):**
   > Martínez-Villaseñor, L., Ponce, H., Brieva, J., Moya-Albor, E., Núñez-Martínez, J., & Peñafort-Asturiano, C. (2019). *UP-Fall Detection Dataset: A Multimodal Approach*. **Sensors**, 19(9), 1988. [https://doi.org/10.3390/s19091988](https://doi.org/10.3390/s19091988)

2. **UP-Fall 3D Skeletons Variant (Zenodo):**
   > UP-Fall 3D Skeletons Dataset (CC BY 4.0): [https://zenodo.org/records/12781431](https://zenodo.org/records/12781431)

3. **UP-Fall Official Project Page:**
   > HAR-UP Multimodal Fall Detection Repository: [https://sites.google.com/up.edu.mx/har-up/](https://sites.google.com/up.edu.mx/har-up/)

4. **MediaPipe Pose Landmarker (Google AI Edge):**
   > Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. Documentation: [https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

5. **PyTorch Deep Learning Framework:**
   > Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

6. **LSTM Sequence Modeling:**
   > Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. **Neural Computation**, 9(8), 1735–1780.

7. **OpenCV & scikit-learn:**
   > OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/) | scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
