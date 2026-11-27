# Experimental Results and Limitations

> **Scientific Integrity Notice:** As of this writing, **no training or evaluation runs have been executed** (verified by the absence of saved model checkpoints in `models/`, training logs or plots in `outputs/`, and downloaded datasets in `data/raw/`). All experimental metric tables and figures below are marked **PENDING** and must only be populated from actual logged outputs after running the pipeline.

---

## 1. Current Verification Status

| Component | Status | Notes |
|-----------|--------|-------|
| Pipeline source code (`src/`, `app/`) | Implemented | Pose extraction, preprocessing, LSTM model, training, evaluation, and inference modules exist |
| Synthetic pipeline training & evaluation | **PENDING** | Has not yet been executed (`python -m src.train --synthetic`) |
| UP-Fall 3D Skeleton dataset download | **PENDING** | `data/raw/` does not yet contain the dataset |
| Real-data training & evaluation | **PENDING** | Blocked on dataset download and training execution |
| Live / real-world validation | **Not tested** | No real-home or clinical testing has been conducted |

---

## 2. Experimental Setup

### Hardware Environment
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM)
- **CPU:** 8 cores
- **System Memory (RAM):** 16 GB

### Software Environment
- **Language:** Python 3.13
- **Deep Learning Framework:** PyTorch (`torch>=2.0.0`)
- **Pose Estimation:** MediaPipe Pose (`mediapipe>=0.10.0`, 33 body landmarks × 4 features `[x, y, z, visibility]` = 132 features/frame)
- **Supporting Libraries:** OpenCV (`opencv-python>=4.8.0`), NumPy (`numpy>=1.24.0`), pandas (`pandas>=2.0.0`), scikit-learn (`scikit-learn>=1.3.0`), Matplotlib (`matplotlib>=3.7.0`)

### Model & Training Configuration (from `src/config.py` and `src/model.py`)
- **Architecture (`FallDetectionLSTM`):** 2-layer stacked LSTM (`hidden_size=128`, `dropout=0.3`), followed by `Dropout(0.3)`, `Linear(128, 32)`, `ReLU`, and `Linear(32, 2)`
- **Input Window:** `SEQUENCE_LENGTH = 30` frames (~1.0 s at 30 FPS), `SEQUENCE_STRIDE = 15` frames (50% overlap)
- **Feature Normalization:** `MinMaxScaler(feature_range=(0, 1))` fitted on training split only
- **Optimization:** Adam (`lr=0.001`), `CrossEntropyLoss`, `BATCH_SIZE = 32`, `NUM_EPOCHS = 50`, `EARLY_STOPPING_PATIENCE = 10`, `RANDOM_SEED = 42`
- **Binary Classes:** `0` = Normal Activity, `1` = Potential Fall

### Dataset
- **Primary Target Dataset:** **UP-Fall Detection Dataset — 3D Skeleton variant** (pending download into `data/raw/`).
  - 17 subjects performing 11 activities (Activities 1–5: 5 fall types; Activities 6–11: 6 daily living activities), 3 trials each.
  - Planned subject-level split (to prevent subject identity leakage): Subjects 1–11 (train, ~65%), Subjects 12–14 (validation, ~17%), Subjects 15–17 (test, ~18%).
- **Pipeline Verification Data:** **Synthetic data** (`src/dataset.py:generate_synthetic_data`), which generates simplified sinusoidal oscillations for normal activity and synthetic linear `y`/`z` shifts for falls.

---

## 3. Results with Synthetic Data

> **Status: To be filled after running training**

> [!WARNING]
> **Scope of Synthetic Results:** Synthetic sequences are generated from simple mathematical patterns (`sin` oscillations vs. artificial downward `y`-coordinate shifts) rather than human biomechanics or real camera noise. Results on synthetic data **only validate that the code pipeline trains, saves artifacts, and computes metrics without error**. They provide **zero evidence** of real-world fall detection accuracy.

### Evaluation Metrics (Synthetic Test Split)

Commands to generate results:
```bash
python -m src.train --synthetic
python -m src.evaluate --synthetic
```

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | _pending_ | Overall classification accuracy on synthetic test split |
| Precision | _pending_ | `pos_label=1` (Potential Fall); reflects false alarm rate |
| Recall | _pending_ | `pos_label=1` (Potential Fall); critical for missed-fall risk |
| F1-Score | _pending_ | Harmonic mean of Precision and Recall |

**Additional Recorded Details (pending):**
- Best epoch / early-stopping epoch: `_pending_`
- Best validation loss: `_pending_`
- Training wall-clock time on GTX 1650 / CPU: `_pending_`

### Confusion Matrix (Synthetic)

<!-- Replace placeholder once `outputs/confusion_matrix.png` is generated from a synthetic run -->
```text
[PENDING: Confusion matrix image will be placed here after running `python -m src.evaluate --synthetic` -> outputs/confusion_matrix.png]
```

### Training and Validation Loss Curves (Synthetic)

<!-- Replace placeholder once `outputs/loss_curves.png` (and `outputs/accuracy_curves.png`) are generated -->
```text
[PENDING: Loss curves image will be placed here after running `python -m src.evaluate --synthetic` -> outputs/loss_curves.png]
```

---

## 4. Results with Real Data (UP-Fall 3D Skeleton)

> **Status: PENDING — requires dataset download and training**

Once the UP-Fall 3D Skeleton dataset is downloaded and placed in `data/raw/`, training and evaluation will be executed on the subject-disjoint test split (Subjects 15–17):
```bash
python -m src.train
python -m src.evaluate
```

### Evaluation Metrics (UP-Fall Test Split — Subjects 15–17)

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | _pending_ | Subject-disjoint test evaluation |
| Precision | _pending_ | `pos_label=1` (Potential Fall) |
| Recall | _pending_ | `pos_label=1` (Potential Fall) |
| F1-Score | _pending_ | Harmonic mean of Precision and Recall |

**Additional Recorded Details (pending):**
- Best epoch / early-stopping epoch: `_pending_`
- Best validation loss (Subjects 12–14): `_pending_`
- Per-class / per-activity breakdown (false positives on lying down vs. jumping vs. picking up objects): `_pending_`
- Training wall-clock time: `_pending_`

### Confusion Matrix (Real Data)

```text
[PENDING: Confusion matrix image from UP-Fall test set -> outputs/confusion_matrix.png]
```

### Training and Validation Loss Curves (Real Data)

```text
[PENDING: Loss curves image from UP-Fall training -> outputs/loss_curves.png]
```

---

## 5. Limitations

This system is an early-stage research prototype. Even after training on UP-Fall, the following structural and methodological limitations apply:

1. **Single-person detection:** MediaPipe Pose (`mp.solutions.pose`) detects at most one person per frame. In multi-person environments (e.g., a caregiver assisting a resident, family members passing through the frame, or pets/clutter causing pose switches), the extractor may track the wrong individual or jump between people across frames, corrupting the 30-frame temporal sequence.
2. **Controlled laboratory environment:** The UP-Fall dataset was recorded in a staged laboratory setting with healthy young volunteers performing scripted activities on protective mats. Real homes have varied floor plans, clutter, rugs, dim corridors, and older adults with differing mobility patterns, gait speeds, and assistive devices (walkers, canes).
3. **Camera angle dependence:** MediaPipe returns normalized image coordinates `(x, y)` and hip-relative depth `z`, which are sensitive to camera elevation, tilt, and viewpoint (front-facing vs. overhead vs. side profile). A model trained on fixed lab camera viewpoints may degrade substantially if deployed at a different mounting angle or distance.
4. **Lighting conditions:** Pose estimation relies on RGB visual appearance. Low-light nighttime conditions (when many residential falls occur), harsh backlighting from windows, or sudden illumination changes reduce landmark accuracy and visibility scores, or cause MediaPipe to fail to detect a person altogether.
5. **Occlusion:** Furniture (beds, sofas, tables, chairs), doorways, blankets, or partial body framing can hide lower-body or upper-body landmarks. When landmarks are occluded, MediaPipe either lowers visibility confidence or hallucinates joint positions, introducing noise into the LSTM input vector.
6. **Limited fall diversity:** The UP-Fall protocol covers only **5 scripted fall types** (falling forward using hands, falling forward using knees, falling backward, falling sideward, and falling while sitting in a chair) alongside 6 normal activities. Real-world falls—such as slow collapses against a wall, slipping in a bathroom, rolling out of bed, or fainting—may have very different kinematics and durations.
7. **Synthetic data limitations:** The built-in synthetic generator (`generate_synthetic_data`) produces artificial sinusoidal oscillations and linear coordinate drops. It does not model articulated joint constraints, camera projection geometry, occlusion, or realistic transition dynamics, and cannot substitute for real-world data.
8. **Not medically or clinically validated:** This software has not undergone clinical trials, regulatory review, or prospective testing with elderly or vulnerable populations. It must not be treated as a medical device or relied upon for life-safety alerting.
9. **Prediction latency (window buffering):** The LSTM requires a full window of `30` frames (approx. 1.0 second at 30 FPS; longer if frame rate drops or pose detection misses frames) before producing its first prediction (`len(landmark_buffer) == 30` in `src/inference.py`). Furthermore, rapid falls shorter than 30 frames or falls occurring at window boundaries may only trigger alerts after impact has already occurred.
10. **No multi-camera fusion:** The current pipeline processes a single monocular video stream. Without multi-view fusion or depth-sensor integration, blind spots, self-occlusion (e.g., falling directly toward or away from the camera), and room-scale coverage remain unresolved.

---

## 6. Possible Sources of Error

1. **MediaPipe landmark detection errors:** Motion blur during fast falls, unusual body orientations (lying horizontally, curled posture), loose clothing, or self-occlusion can cause MediaPipe Pose to misplace joints, swap left/right limbs, or drop detections (`pose_landmarks is None`), creating missing frames or erratic spikes in the 132-feature vector.
2. **Data leakage if not split by subject:** Random frame-level or sequence-level splitting leaks subject-specific body proportions, movement habits, and overlapping sliding windows (`stride=15` creates 50% frame overlap between adjacent sequences) across train and test sets, artificially inflating accuracy. While `src/config.py` specifies a subject-disjoint split (Subjects 1–11 / 12–14 / 15–17), any future modification or external dataset integration that breaks subject-level grouping would invalidate evaluation results.
3. **Class imbalance:** In continuous monitoring—and depending on trial durations in datasets—normal daily activities vastly outnumber fall events. Note that `src/train.py` currently uses unweighted `nn.CrossEntropyLoss()`, so severe class imbalance in real data can bias the model toward predicting "Normal Activity," raising accuracy while degrading **fall recall** (increasing dangerous false negatives).
4. **Overfitting on a small dataset:** UP-Fall contains only 17 subjects (11 training subjects). With 132 input features per frame and a two-layer LSTM, the model can easily memorize subject-specific trajectories, background/camera quirks, or scripted start/end cues unless regularized and evaluated strictly on unseen subjects.
5. **Domain shift (lab dataset to real world):** Differences in camera intrinsics, frame rates, clothing, subject age/mobility, and daily routines between UP-Fall and a target deployment environment create substantial domain shift. In particular, non-fall activities such as lying down quickly, sitting heavily, or bending to pick up objects (Activities 8, 9, 11) closely resemble fall kinematics and are primary sources of false positives.

---

## 7. Comparison with Existing Work

Published vision- and sensor-based fall detection systems generally follow one of several paradigms:

| Approach | Typical Inputs | Strengths | Weaknesses |
|----------|---------------|-----------|------------|
| **Wearable IMU (accelerometers / gyroscopes)** | 3-axis acceleration & angular velocity | Direct impact/orientation signal; privacy-preserving; works anywhere | Requires user compliance (charging, wearing device consistently) |
| **Ambient / Radar / Wi-Fi CSI** | Millimeter-wave radar, Wi-Fi channel state, floor vibration | Contactless; works in darkness; preserves visual privacy | Hardware cost; sensitive to room layout and multi-person interference |
| **Raw RGB / Optical Flow CNNs (2D/3D CNNs)** | Video frames or optical flow stacks | Captures rich appearance & motion cues directly | High compute/VRAM cost; sensitive to background/appearance overfitting; severe privacy concerns |
| **Skeleton-based Temporal Models (GCN / LSTM / Transformer)** | 2D/3D pose keypoints (e.g., OpenPose, MediaPipe, AlphaPose) | Lightweight; background-invariant; interpretable joint trajectories | Completely dependent on upstream pose estimator accuracy; loses context when person is occluded |
| **This Prototype (MediaPipe + 2-Layer LSTM)** | 33 3D BlazePose landmarks × 4 values over 30 frames | Runs on modest consumer GPU (GTX 1650 4 GB); modular codebase | Single-person, single-camera, no class-weighting or spatial-graph modeling; **untested on real data so far** |

**Methodological Note on Fair Comparison:** Direct numerical comparisons against published UP-Fall benchmarks (such as the original multimodal UP-Fall study by Martínez-Villaseñor et al., 2019, or subsequent skeleton-based GCN/Transformer papers) are only scientifically valid when evaluated under **identical conditions**: the same modality subset, the exact same subject-level split or cross-validation folds, the same sliding-window stride and labeling rule, and the same metric definitions (e.g., sequence-level vs. event-level recall and false-alarm rate per hour). Because this repository has not yet completed training/evaluation on UP-Fall, no benchmark comparison claims can be made at this stage.

---

## 8. Ethical Considerations

1. **Privacy in continuous video monitoring:** Placing cameras in living spaces—especially bedrooms and bathrooms where falls frequently occur—creates severe privacy risks. Although this pipeline converts RGB frames into 132-dimensional skeleton vectors before LSTM classification, the pose-extraction stage still requires capturing raw camera frames in memory. Any deployment should process video strictly on-device (edge inference) and discard raw frames immediately without saving or transmitting video.
2. **Informed consent and autonomy:** Continuous monitoring must never be deployed covertly. Residents, patients, and any visitors or caregivers entering the monitored space must be clearly informed about when and where camera-based sensing is active, what data is retained, and how to pause or opt out.
3. **Data storage, retention, and security:** Even anonymized pose coordinates or fall-event logs constitute sensitive health-related behavioral data (revealing daily schedules, mobility decline, and occupancy). Saved outputs, logs, and model artifacts should follow data minimization principles, use encryption at rest and in transit if alerts are sent off-device, and have clear retention limits.
4. **Not a replacement for human caregivers:** Automated fall detection is an assistive alerting aid, not a substitute for human care, clinical assessment, or emergency response protocols. Because both false negatives (missed falls) and false positives (alarm fatigue) can occur, any real-world deployment would require human-in-the-loop verification and redundant safety mechanisms.

---

## 9. Next Steps to Complete Empirical Evaluation

1. Download the UP-Fall 3D Skeleton dataset into `data/raw/` (see [data/README.md](../data/README.md)).
2. Run and record the synthetic pipeline check (`python -m src.train --synthetic` and `python -m src.evaluate --synthetic`) to verify end-to-end execution and populate Section 3.
3. Run subject-disjoint training and evaluation on UP-Fall (`python -m src.train` and `python -m src.evaluate`) and populate Section 4 with the actual `outputs/evaluation_metrics.json`, `outputs/confusion_matrix.png`, and `outputs/loss_curves.png` results.
4. Inspect per-activity error rates (especially Activities 8, 9, and 11 vs. Activities 1–5) and evaluate class-weighted loss or threshold tuning to prioritize high fall recall.
