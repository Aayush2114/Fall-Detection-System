# Technical Concepts Guide — Fall Detection System (MediaPipe + LSTM)

Welcome! This guide explains every major technical concept used in our **Fall Detection System** in plain, beginner-friendly language written for a 2nd-year Computer Science student.

Each section follows a consistent structure:
- **Definition** — What the concept means (with an everyday analogy)
- **Why It Is Used** — What problem it solves
- **How It Is Used in This Project** — Where and how it appears in our codebase
- **Python Example** — A small, runnable Python snippet demonstrating the idea

---

## Table of Contents

1. [Computer Vision](#1-computer-vision)
2. [OpenCV](#2-opencv)
3. [Human Pose Estimation](#3-human-pose-estimation)
4. [MediaPipe](#4-mediapipe)
5. [Landmarks](#5-landmarks)
6. [Feature Vector](#6-feature-vector)
7. [Normalization (Min-Max Scaling)](#7-normalization-min-max-scaling)
8. [Sequential Data](#8-sequential-data)
9. [Sliding Window](#9-sliding-window)
10. [Neural Network Basics](#10-neural-network-basics)
11. [Recurrent Neural Network (RNN)](#11-recurrent-neural-network-rnn)
12. [LSTM (Long Short-Term Memory)](#12-lstm-long-short-term-memory)
13. [CNN vs LSTM](#13-cnn-vs-lstm)
14. [PyTorch](#14-pytorch)
15. [Tensor](#15-tensor)
16. [Loss Function (CrossEntropyLoss)](#16-loss-function-crossentropyloss)
17. [Backpropagation](#17-backpropagation)
18. [Optimizer (Adam)](#18-optimizer-adam)
19. [Overfitting](#19-overfitting)
20. [Dropout](#20-dropout)
21. [Batch Size](#21-batch-size)
22. [Epoch](#22-epoch)
23. [Train/Validation/Test Split](#23-trainvalidationtest-split)
24. [Evaluation Metrics](#24-evaluation-metrics)
25. [Data Leakage](#25-data-leakage)

---

## 1. Computer Vision

### Definition
**Computer Vision (CV)** is a field of Artificial Intelligence that teaches computers how to understand and interpret images and videos.

**Analogy:** When you look at a photo, your brain instantly recognizes "a person sitting on a chair." A computer, however, only sees a giant grid of numbers representing pixel brightness and color. Computer Vision is the set of techniques that turns that grid of numbers into meaningful understanding.

For a color image of size `480 × 640` pixels with 3 color channels (Red, Green, Blue), the computer sees a 3D array of shape `(480, 640, 3)` where each number is typically between `0` and `255`.

### Why It Is Used
Cameras are cheap, non-intrusive sensors. Unlike wearable fall-detection pendants (which an elderly person might forget to wear or charge), a camera can monitor a room passively. Computer Vision allows software to analyze video frames automatically without a human watching the screen 24/7.

### How It Is Used in This Project
Instead of feeding raw pixels straight into a heavy video classifier, our system uses Computer Vision in two stages:
1. Read raw video frames (`src/pose_extraction.py` and `src/inference.py`).
2. Locate the human body and extract joint coordinates (`mediapipe` Pose), turning complex images into compact numerical pose data.

### Python Example
```python
import numpy as np

# A tiny 3x3 RGB image (height=3, width=3, channels=3)
# Values range from 0 (dark) to 255 (bright)
tiny_image = np.array([
    [[255, 0, 0],   [0, 255, 0],   [0, 0, 255]],    # Red, Green, Blue pixels
    [[255, 255, 0], [255, 255, 255], [0, 0, 0]],    # Yellow, White, Black
    [[128, 128, 128], [50, 100, 150], [200, 50, 50]]
], dtype=np.uint8)

print("Image shape (H, W, Channels):", tiny_image.shape)
print("Top-left pixel RGB:", tiny_image[0, 0])
```

---

## 2. OpenCV

### Definition
**OpenCV** (Open Source Computer Vision Library, imported in Python as `cv2`) is the most widely used library for reading, writing, processing, and displaying images and videos.

**Analogy:** If raw video files (`.mp4`, `.avi`) and webcam streams are locked boxes of images, OpenCV is the Swiss Army knife that opens them, hands you one frame at a time, lets you draw boxes/text on them, and displays them on screen.

A key quirk of OpenCV is **color channel order**:
- Most libraries (including **MediaPipe** and **Matplotlib**) expect **RGB** (Red, Green, Blue).
- **OpenCV** reads images in **BGR** (Blue, Green, Red) for historical reasons.
If you pass a BGR image directly to a model expecting RGB, reds and blues get swapped (skin looks bluish!), hurting accuracy.

### Why It Is Used
Python does not have built-in tools to decode `.mp4` video codecs, connect to webcams, convert BGR to RGB, or draw skeletons and alert banners on frames. OpenCV handles all of this quickly in C++ under the hood.

### How It Is Used in This Project
In `src/pose_extraction.py` and `src/inference.py`, OpenCV is used to:
- Open video files (`cv2.VideoCapture(video_path)`) or webcams (`cv2.VideoCapture(0)`).
- Read frames in a loop (`ret, frame = cap.read()`).
- Convert color space before MediaPipe runs (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`).
- Draw prediction overlays (`cv2.rectangle`, `cv2.putText`), show live windows (`cv2.imshow`), and save annotated videos (`cv2.VideoWriter`).
- Notice in `src/config.py`, visualization colors like `NORMAL_COLOR = (0, 255, 0)` (Green) and `FALL_COLOR = (0, 0, 255)` (Red) are written in **BGR** format because OpenCV draws them.

### Python Example
```python
import cv2
import numpy as np

# Create a blank 200x300 BGR image (all zeros = black)
bgr_image = np.zeros((200, 300, 3), dtype=np.uint8)

# In BGR, pure Red is (Blue=0, Green=0, Red=255)
bgr_image[:, :] = (0, 0, 255)

# Convert BGR -> RGB before passing to libraries like MediaPipe
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

print("BGR pixel value:", bgr_image[0, 0])  # [0, 0, 255]
print("RGB pixel value:", rgb_image[0, 0])  # [255, 0, 0]

# Reading a video file with VideoCapture (standard pattern)
# cap = cv2.VideoCapture("sample.mp4")  # or 0 for webcam
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# cap.release()
```

---

## 3. Human Pose Estimation

### Definition
**Human Pose Estimation** is the Computer Vision task of locating key human body joints (such as nose, shoulders, elbows, wrists, hips, knees, and ankles) in an image or video and connecting them into a "stick figure" skeleton.

**Analogy:** Think of motion-capture suits with ping-pong balls used in movies and video games. Human Pose Estimation achieves a similar result *without* any special suit—just using a regular camera and a machine learning model.

### Why It Is Used
Raw video frames contain lots of irrelevant background details: wall color, furniture, lighting, and clothing. If you train a model on raw video pixels, it might mistakenly memorize the room rather than the motion. Human Pose Estimation strips away the background and clothing, leaving only the geometry and motion of the person's body.

### How It Is Used in This Project
Instead of classifying raw video images, we first run Human Pose Estimation on every frame (`extract_landmarks_from_frame` in `src/pose_extraction.py`) to turn each video frame into a clean skeleton of 33 body joints. Our downstream LSTM classifier only ever looks at these joint positions—making the system lightweight, fast, and privacy-friendly.

### Python Example
```python
# Conceptual representation of pose estimation output for 1 frame
pose_joints = {
    "nose": (0.50, 0.18),
    "left_shoulder": (0.42, 0.30),
    "right_shoulder": (0.58, 0.30),
    "left_hip": (0.45, 0.55),
    "right_hip": (0.55, 0.55),
}

# We can compute simple geometric cues, like torso midpoint height
mid_shoulder_y = (pose_joints["left_shoulder"][1] + pose_joints["right_shoulder"][1]) / 2
mid_hip_y = (pose_joints["left_hip"][1] + pose_joints["right_hip"][1]) / 2
print(f"Shoulder Y: {mid_shoulder_y:.2f}, Hip Y: {mid_hip_y:.2f}")
```

---

## 4. MediaPipe

### Definition
**MediaPipe** is an open-source, cross-platform Machine Learning framework developed by Google for building real-time perception pipelines (such as face mesh, hand tracking, and human pose estimation). Its **Pose** solution (`mp.solutions.pose`) detects **33 full-body 3D landmarks** from a single RGB image in real time, even on a standard CPU.

**Analogy:** Training your own pose detector from scratch would require millions of labeled images and days on expensive GPUs. MediaPipe is like hiring an already-trained expert who immediately points out all 33 body joints on every frame you hand them.

### Why It Is Used
- **Fast on CPU:** Runs at real-time speeds (~30 FPS) without needing a high-end GPU.
- **Built-in tracking:** When `static_image_mode=False`, MediaPipe tracks the person from frame to frame instead of re-detecting from scratch every frame, giving smoother and faster results.
- **Rich 33-point skeleton:** Covers head, torso, arms, hands, legs, and feet.

### How It Is Used in This Project
In `src/pose_extraction.py` and `src/inference.py`, we initialize MediaPipe Pose as:
```python
mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```
These thresholds (`MIN_DETECTION_CONFIDENCE = 0.5`, `MIN_TRACKING_CONFIDENCE = 0.5`, `NUM_LANDMARKS = 33`) are defined in `src/config.py`. MediaPipe also provides `mp.solutions.drawing_utils` to draw the skeleton onto video frames.

### Python Example
```python
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Create a dummy blank RGB image (480x640) to test MediaPipe initialization
dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    results = pose.process(dummy_rgb)
    if results.pose_landmarks is None:
        print("MediaPipe ran successfully (no person in blank image).")
    else:
        print("Detected landmarks:", len(results.pose_landmarks.landmark))
```

---

## 5. Landmarks

### Definition
**Landmarks** are the specific 3D points (body joints) detected by MediaPipe Pose. Each of the 33 landmarks contains **4 numbers**:
- **`x`**: Horizontal position, normalized by image width (`0.0` = left edge, `1.0` = right edge).
- **`y`**: Vertical position, normalized by image height (`0.0` = top edge, `1.0` = bottom edge). *Note: larger `y` means lower down in the image (closer to the floor).*
- **`z`**: Relative depth (distance toward/away from the camera), centered around the midpoint of the hips. Negative `z` means closer to the camera than the hips; roughly on the same scale as `x`.
- **`visibility`**: A confidence score in `[0.0, 1.0]` estimating how likely the joint is visible (not hidden/occluded).

**Normalized coordinates** means `x` and `y` are expressed as fractions of the image width/height rather than raw pixels. For example, `x = 0.5, y = 0.5` is the exact center of the frame whether the video is `640×480` or `1920×1080`.

**Analogy:** Imagine giving someone directions on a football field using percentages ("standing 50% across the width and 80% down the length") instead of exact meters—it works regardless of how big the stadium screen is.

### Why It Is Used
Normalized landmarks make the system resolution-independent and give us exact geometric signals for falls (for instance, during a fall, the `y` values of the head, shoulders, and hips increase rapidly toward the bottom of the frame).

### How It Is Used in This Project
In `src/pose_extraction.py` (`extract_landmarks_from_frame`), we loop over `results.pose_landmarks.landmark` (all 33 joints, such as `0: Nose`, `11/12: Shoulders`, `23/24: Hips`, `25/26: Knees`, `27/28: Ankles`) and collect `[lm.x, lm.y, lm.z, lm.visibility]` for each one. If a person is briefly missed in some frames, `interpolate_missing_frames()` in `src/preprocessing.py` fills the gaps using linear interpolation.

### Python Example
```python
# Simulating a single MediaPipe landmark (e.g., landmark 0 = Nose)
nose_landmark = {
    "x": 0.52,          # Slightly right of horizontal center
    "y": 0.20,          # Near top 20% of frame (standing)
    "z": -0.15,         # Slightly closer to camera than hips
    "visibility": 0.98  # 98% confident this joint is visible
}

frame_width, frame_height = 1280, 720
pixel_x = int(nose_landmark["x"] * frame_width)
pixel_y = int(nose_landmark["y"] * frame_height)

print(f"Normalized: ({nose_landmark['x']}, {nose_landmark['y']})")
print(f"Pixel coordinates on 1280x720 video: ({pixel_x}, {pixel_y})")
```

---

## 6. Feature Vector

### Definition
A **Feature Vector** is a flat 1D list (or array) of numbers that represents a single data item so a machine learning model can process it.

**Analogy:** When you fill out a medical form (age, height, weight, heart rate), all your answers together form a row of numbers describing you—that row is your feature vector.

In our project, MediaPipe outputs 33 landmarks, and each landmark has 4 values (`x, y, z, visibility`). Flattening them into one 1D vector gives:
$$33 \text{ landmarks} \times 4 \text{ values} = 132 \text{ features per frame}$$

### Why It Is Used
Neural networks cannot read nested objects like `results.pose_landmarks.landmark[11].x`. They require clean, fixed-size numeric arrays where every position (index `0` to `131`) always has the same meaning.

### How It Is Used in This Project
In `src/config.py`:
- `NUM_LANDMARKS = 33`
- `LANDMARK_FEATURES = 4`
- `NUM_FEATURES = 132`

In `src/pose_extraction.py`, each frame's 33 landmarks are flattened into a NumPy array of shape `(132,)`. A video with `N` frames becomes a 2D array of shape `(N, 132)`.

### Python Example
```python
import numpy as np

# Suppose we have 33 landmarks, each with 4 values: (x, y, z, visibility)
landmarks_2d = np.random.rand(33, 4).astype(np.float32)

# Flatten into a 1D feature vector of length 132
feature_vector = landmarks_2d.flatten()

print("2D landmarks shape:", landmarks_2d.shape)       # (33, 4)
print("1D feature vector shape:", feature_vector.shape) # (132,)
```

---

## 7. Normalization (Min-Max Scaling)

### Definition
**Normalization (Min-Max Scaling)** rescales numerical features so they all lie within the same range, typically `[0, 1]`.

The formula for each feature $x$ is:
$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Analogy:** Imagine comparing a student's test score out of 100 (`85/100`) with another exam out of 10 (`9/10`). To compare them fairly, you convert both to the same scale (`0.85` and `0.90`).

### Why It Is Used
Our 132 features live on slightly different numerical ranges: `x`, `y`, and `visibility` are in `[0, 1]`, while `z` (depth) can be negative or larger than `1` (or vary across datasets like UP-Fall 3D skeleton coordinates). If one feature has values from `-500` to `+500` and another is `0` to `1`, the neural network's gradients get skewed and training becomes slow or unstable. Scaling everything to `[0, 1]` keeps learning smooth and balanced.

### How It Is Used in This Project
In `src/preprocessing.py`, `normalize_features()` uses `sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`:
- On **training data**, we call `scaler.fit_transform(features)` (`fit=True`) to learn $x_{\min}$ and $x_{\max}$ from training frames only, and save the fitted scaler (`feature_scaler.pkl` / `scaler.pkl`).
- On **validation, test, and live inference data**, we call `scaler.transform(features)` (`fit=False`) using the *already-fitted* training scaler so new data is scaled identically without **data leakage**.

### Python Example
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Raw sample features: e.g., x in [0, 1], z in [-2.0, 2.0]
train_data = np.array([
    [0.2, -2.0],
    [0.5,  0.0],
    [0.8,  2.0],
], dtype=np.float32)

test_data = np.array([
    [0.5, 1.0],
], dtype=np.float32)

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)  # Fit ONLY on train
test_scaled = scaler.transform(test_data)        # Transform test using train min/max

print("Scaled train:\n", train_scaled)
print("Scaled test:\n", test_scaled)
```

---

## 8. Sequential Data

### Definition
**Sequential Data** (also called **time-series data** when indexed by time) is data where the **order of elements matters**. Unlike **static data** (like a single photo or a spreadsheet row of house prices), shuffling the order of sequential data destroys its meaning.

**Analogy:** Compare a single still photo of someone mid-air (static data) with a 1-second video clip (sequential data). From a single frame, someone bending down to tie their shoe and someone falling forward can look nearly identical! Only by watching the *order* of frames—`standing -> tilting rapidly -> lying on floor`—can you tell it is a fall (as opposed to `lying on floor -> pushing up -> standing`).

### Why It Is Used
A fall is a **temporal event** (an action unfolding over time), not a static posture. Treating frames independently misses speed, direction, and transitions.

### How It Is Used in This Project
Each input sample passed to our LSTM model in `src/dataset.py` and `src/model.py` is not a single `(132,)` frame vector, but a time sequence of `SEQUENCE_LENGTH = 30` consecutive frames—a tensor of shape `(30, 132)`. At 30 FPS, 30 frames represent 1 full second of movement.

### Python Example
```python
import numpy as np

# Static data: 1 frame of 4 numbers (order across samples doesn't matter)
single_frame = np.array([0.5, 0.4, -0.1, 0.99])

# Sequential data: 5 frames over time tracking a joint's y-coordinate
# Increasing y means moving downward toward the floor
standing_to_falling_y = np.array([0.20, 0.25, 0.42, 0.68, 0.85])
getting_up_y          = np.array([0.85, 0.68, 0.42, 0.25, 0.20])

print("Fall delta (end - start):", standing_to_falling_y[-1] - standing_to_falling_y[0])
print("Stand-up delta (end - start):", getting_up_y[-1] - getting_up_y[0])
```

---

## 9. Sliding Window

### Definition
A **Sliding Window** is a technique for slicing a long, continuous stream of frames into fixed-length, often overlapping chunks (sequences) using two parameters:
- **Window / Sequence Length ($L$)**: How many consecutive frames are in each chunk.
- **Stride ($S$)**: How many frames we step forward before starting the next chunk.

```text
Continuous frames: [f0, f1, f2, f3, f4, f5, f6, ... f29, f30, ... f44, f45, ...]

Sequence 0 (0..29):   [========== 30 frames ==========]
Sequence 1 (15..44):             [========== 30 frames ==========]
                      <--------->
                      stride = 15 (50% overlap)
```

**Analogy:** Imagine reading a long ticker-tape through a cardboard cutout that only shows 30 letters at a time. You read those 30 letters, slide the cutout forward by 15 letters, read the next 30, and repeat.

### Why It Is Used
1. Neural networks (like PyTorch batches) need fixed-size sequences, while videos have varying lengths.
2. Using `stride < sequence_length` (overlapping windows) creates **more training samples** from limited video data and ensures a fall that starts halfway through one window is cleanly captured in the next window.
3. During live webcam inference, a sliding buffer lets the system make continuous predictions every new frame without waiting for the entire video to finish.

### How It Is Used in This Project
- **Training/Preprocessing (`src/config.py`, `src/preprocessing.py`):** `create_sequences()` uses `SEQUENCE_LENGTH = 30` and `SEQUENCE_STRIDE = 15` (50% overlap), turning `(N, 132)` frame arrays into `(num_sequences, 30, 132)`. If a clip has fewer than 30 frames, it pads with zeros.
- **Real-time Inference (`src/inference.py`):** Uses Python's `collections.deque(maxlen=30)` as a rolling buffer—each new frame pushes out the oldest frame once 30 frames are collected.

### Python Example
```python
import numpy as np

def sliding_window(data, seq_length=4, stride=2):
    sequences = []
    for start in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[start : start + seq_length])
    return np.array(sequences)

frames = np.arange(10).reshape(10, 1)  # 10 frames, 1 feature each
windows = sliding_window(frames, seq_length=4, stride=2)

print("Windows shape:", windows.shape)   # (4, 4, 1)
print("First window:", windows[0].ravel())
print("Second window:", windows[1].ravel())
```

---

## 10. Neural Network Basics

### Definition
An **Artificial Neural Network (ANN)** is a computing model inspired by neurons in the brain. It is built from layers of simple mathematical units (**neurons**). Each neuron computes:
$$y = f\left(\sum_{i} w_i x_i + b\right)$$
- **Weights ($w_i$)**: Numbers that control how strongly each input matters (learned during training).
- **Bias ($b$)**: An adjustable offset added to shift the output (also learned).
- **Activation Function ($f$)**: A non-linear function (like **ReLU**: $\max(0, z)$) applied to the result.

**Analogy:** Think of deciding whether to carry an umbrella: you weigh different clues (`dark clouds × 0.8 + weather forecast × 0.9 + wind × 0.2`), add your general caution level (`bias`), and make a threshold decision (`activation`).

### Why It Is Used
Without non-linear activation functions (like `ReLU`), stacking 10 linear layers would mathematically collapse into a single linear equation, unable to learn complex movement patterns. Weights, biases, and activations let neural networks approximate complex decision boundaries from data.

### How It Is Used in This Project
In `src/model.py`, after the LSTM extracts temporal features, our **classification head** uses standard neural network layers:
- `self.fc1 = nn.Linear(128, 32)` — 32 neurons, each computing a weighted sum + bias over 128 inputs.
- `self.relu = nn.ReLU()` — non-linear activation (`max(0, x)`).
- `self.fc2 = nn.Linear(32, 2)` — 2 output neurons producing raw class scores (logits) for `Normal Activity` and `Potential Fall`.

### Python Example
```python
import numpy as np

# One artificial neuron with 3 inputs
inputs = np.array([0.8, 0.2, 0.5])
weights = np.array([0.6, -0.4, 0.9])
bias = -0.1

# Weighted sum + bias
z = np.dot(inputs, weights) + bias

# ReLU activation function: max(0, z)
output = max(0.0, z)

print(f"Linear sum z = {z:.3f}")
print(f"After ReLU   = {output:.3f}")
```

---

## 11. Recurrent Neural Network (RNN)

### Definition
A **Recurrent Neural Network (RNN)** is a neural network built specifically for sequential data. Unlike a regular network that treats each input independently, an RNN processes one time step at a time and passes a **hidden state ($h_t$)**—a summary of everything it has seen so far—from step $t-1$ to step $t$.

```text
Frame 1 -> [ RNN ] --(h1)--> [ RNN ] --(h2)--> ... --> [ RNN ] --(h30)--> Prediction
             ^                 ^                         ^
           (h0=0)           Frame 2                   Frame 30
```

**Analogy:** When reading a sentence word by word, you don't forget word 1 when you reach word 5. Your short-term memory (`hidden state`) carries the context forward so the whole sentence makes sense.

### Why It Is Used
A standard feed-forward layer expects a fixed, unordered vector and has no built-in notion of "what happened one frame earlier." An RNN naturally models step-by-step progression across time.

### How It Is Used in This Project
Understanding standard RNNs is the stepping stone to our model: we need a recurrent architecture that reads 30 pose frames in order and summarizes the motion in its final hidden state. In practice, we use an upgraded RNN variant—**LSTM**—because basic RNNs suffer from the **vanishing gradient problem** (forgetting earlier frames in a 30-frame sequence).

### Python Example
```python
import numpy as np

# Toy RNN step: h_t = tanh(W_x * x_t + W_h * h_{t-1})
W_x, W_h = 0.5, 0.8
h = 0.0  # Initial hidden state (memory)

sequence = [0.1, 0.2, 0.7, 0.9]  # e.g., y-position over 4 frames
for t, x_t in enumerate(sequence):
    h = np.tanh(W_x * x_t + W_h * h)
    print(f"Step {t}: input={x_t:.1f}, hidden_state={h:.4f}")
```

---

## 12. LSTM (Long Short-Term Memory)

### Definition
An **LSTM (Long Short-Term Memory)** network is an advanced type of RNN designed to remember important information across long sequences without forgetting early steps.

Inside each LSTM unit, there are **two memory paths** and **three gates** (like valves controlling information flow):
1. **Cell State ($C_t$)**: The "long-term memory conveyor belt" running straight through the sequence, making it easy to carry important facts from frame 1 to frame 30.
2. **Hidden State ($h_t$)**: The working "short-term memory" / output at step $t$.
3. **Forget Gate ($f_t$)**: Decides what old information in the cell state is no longer useful and should be erased.
4. **Input Gate ($i_t$)**: Decides what new information from the current frame should be stored in the cell state.
5. **Output Gate ($o_t$)**: Decides what part of the cell state should be exposed as the current hidden state $h_t$.

**Why better than a plain RNN?** Plain RNNs multiply gradients repeatedly across time steps, causing earlier frames' learning signals to shrink toward zero (**vanishing gradient**). The LSTM's cell state adds/updates information gently via gates, preserving long-range memory.

**Analogy:** Imagine taking notes during a lecture (`Cell State`). The **Forget Gate** erases outdated scratch notes, the **Input Gate** writes down important new formulas, and the **Output Gate** summarizes what you need right now to answer a question.

### Why It Is Used
A fall takes roughly 0.5 to 2 seconds (15–60 frames). To distinguish a genuine fall from sitting down quickly, the model must remember how the person started at frame 1 (`standing upright`) and compare it to frame 30 (`on the floor`). LSTM handles this 30-frame memory reliably.

### How It Is Used in This Project
In `src/model.py`, `FallDetectionLSTM` uses a **2-layer stacked PyTorch LSTM** (`nn.LSTM(input_size=132, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)`):
- It reads `(batch_size, 30, 132)`.
- We take the **final time step's output** (`lstm_out[:, -1, :]`, shape `(batch_size, 128)`), which encodes the entire 1-second motion pattern, and pass it to the linear classifier.

### Python Example
```python
import torch
import torch.nn as nn

# Create a 2-layer LSTM matching our project's input/hidden dimensions
lstm = nn.LSTM(input_size=132, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)

# Dummy batch: 4 sequences, each 30 frames long, 132 features per frame
x = torch.randn(4, 30, 132)

lstm_out, (h_n, c_n) = lstm(x)
last_step_features = lstm_out[:, -1, :]  # Summary after frame 30

print("Full LSTM output shape:", lstm_out.shape)         # (4, 30, 128)
print("Final hidden state h_n shape:", h_n.shape)        # (2, 4, 128)
print("Last time step features:", last_step_features.shape)  # (4, 128)
```

---

## 13. CNN vs LSTM

### Definition
- **CNN (Convolutional Neural Network)**: Uses sliding spatial filters (kernels) to detect visual patterns like edges, textures, shapes, and objects in 2D images or grids.
- **LSTM (Long Short-Term Memory)**: Processes ordered sequences step by step to learn how features evolve over time.

| Aspect | CNN | LSTM |
| :--- | :--- | :--- |
| **Best at** | Spatial patterns (images, object detection) | Temporal patterns (motion, text, time series) |
| **Input shape** | `(Batch, Channels, Height, Width)` | `(Batch, Time_Steps, Features)` |
| **Understands time order?** | Not natively (unless using 3D convolutions) | Yes, built specifically for sequence order |
| **Compute cost on video** | High if processing raw video frames directly | Very low when fed MediaPipe pose vectors |

**Analogy:** A **CNN** is an art critic examining *what is inside a single painting*. An **LSTM** is a sports commentator watching *how a play unfolds second by second*.

### Why It Is Used
Choosing the right architecture avoids wasted compute and improves generalization. CNNs are great at finding where a person's joints are inside each image (in fact, MediaPipe itself uses lightweight CNNs internally to detect the 33 landmarks!). Once we already have the 33 joint coordinates over 30 frames, the remaining problem is purely **temporal motion classification**, which is where LSTM shines.

### How It Is Used in This Project
Our pipeline combines the strengths of both worlds:
1. **MediaPipe Pose** (CNN-based vision model) solves the spatial problem: *image $\rightarrow$ 132 pose numbers*.
2. **FallDetectionLSTM** (`src/model.py`) solves the temporal problem: *30 frames of 132 pose numbers $\rightarrow$ Normal vs Fall*.

### Python Example
```python
import torch
import torch.nn as nn

# CNN expects spatial grid: (batch, channels, height, width)
cnn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
fake_image_batch = torch.randn(4, 3, 64, 64)
print("CNN output shape:", cnn(fake_image_batch).shape)  # (4, 16, 62, 62)

# LSTM expects sequence: (batch, time_steps, features)
lstm = nn.LSTM(input_size=132, hidden_size=64, batch_first=True)
fake_pose_batch = torch.randn(4, 30, 132)
lstm_out, _ = lstm(fake_pose_batch)
print("LSTM output shape:", lstm_out.shape)              # (4, 30, 64)
```

---

## 14. PyTorch

### Definition
**PyTorch** is an open-source deep learning framework (originally created by Meta AI) used to build, train, and run neural networks in Python. Its four core pillars are:
1. **Tensors**: GPU-capable multi-dimensional arrays (like NumPy arrays on steroids).
2. **`nn.Module`**: The base class for building neural network layers and custom models (defining layers in `__init__` and computation in `forward()`).
3. **Forward Pass**: Running input data through `model(x)` to get predictions.
4. **Autograd**: PyTorch's automatic differentiation engine—it tracks every math operation during the forward pass so `.backward()` can automatically compute gradients for training.

**Analogy:** If building a neural network in pure math/NumPy is like building a car engine from raw metal, PyTorch is a modular automotive toolkit with pre-built parts (`nn.LSTM`, `nn.Linear`) and an automated assembly line (`autograd`).

### Why It Is Used
PyTorch makes deep learning code readable, Pythonic, easy to debug, and able to switch between CPU and GPU (`cuda`) with a single `.to(device)` call.

### How It Is Used in This Project
- `src/model.py`: `FallDetectionLSTM` inherits from `nn.Module`, defines layers (`nn.LSTM`, `nn.Dropout`, `nn.Linear`, `nn.ReLU`), and implements `forward(self, x)`.
- `src/dataset.py`: Uses `torch.utils.data.Dataset` and `DataLoader` to feed data.
- `src/train.py`: Uses `autograd` (`loss.backward()`), `nn.CrossEntropyLoss()`, and `torch.optim.Adam`.
- `src/inference.py`: Uses `torch.no_grad()` and `model.eval()` for fast predictions.

### Python Example
```python
import torch
import torch.nn as nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)

model = TinyClassifier()
x = torch.randn(2, 4, requires_grad=True)
out = model(x)            # Forward pass
loss = out.sum()
loss.backward()           # Autograd computes gradients automatically
print("Output shape:", out.shape)
print("Input gradient shape:", x.grad.shape)
```

---

## 15. Tensor

### Definition
A **Tensor** is a multi-dimensional array of numbers—the fundamental data structure in PyTorch.
- **0D Tensor (Scalar)**: A single number, e.g., `loss = 0.42`, shape `()`.
- **1D Tensor (Vector)**: A list of numbers, e.g., `[0, 1, 1, 0]` (batch labels), shape `(4,)`.
- **2D Tensor (Matrix)**: A table of numbers, e.g., one sequence of 30 frames × 132 features, shape `(30, 132)`.
- **3D Tensor**: A stack of matrices, e.g., a batch of 32 sequences, shape `(32, 30, 132)`.

**Analogy:** A scalar is a single page number, a 1D vector is a line of text, a 2D matrix is one full page, and a 3D tensor is a whole book of pages.

### Why It Is Used
Tensors allow PyTorch to perform millions of matrix multiplications in parallel (especially on GPUs) and carry metadata needed for `autograd` (gradients) and data types (`float32` for features, `int64`/`torch.long` for class labels).

### How It Is Used in This Project
- Single frame features: shape `(132,)`
- One sequence sample (`FallDetectionDataset.__getitem__`): shape `(30, 132)` (`dtype=torch.float32`)
- One training batch (`DataLoader` with `batch_size=32`):
  - `sequences`: shape `(32, 30, 132)`
  - `labels`: shape `(32,)` (`dtype=torch.long`)
- Model output logits: shape `(32, 2)`
- Single live inference sample (`src/inference.py`): `.unsqueeze(0)` adds a batch dimension of 1 so `(30, 132)` becomes `(1, 30, 132)`.

### Python Example
```python
import torch

scalar = torch.tensor(1)                      # 0D: shape ()
labels = torch.tensor([0, 1, 0], dtype=torch.long)  # 1D: shape (3,)
one_sequence = torch.randn(30, 132)           # 2D: shape (30, 132)
batch_sequences = torch.randn(32, 30, 132)    # 3D: shape (32, 30, 132)

# Adding a batch dimension for single-sample inference
single_inference_input = one_sequence.unsqueeze(0)

print("0D shape:", scalar.shape)
print("1D shape:", labels.shape)
print("2D shape:", one_sequence.shape)
print("3D batch shape:", batch_sequences.shape)
print("Unsqueezed shape:", single_inference_input.shape)
```

---

## 16. Loss Function (CrossEntropyLoss)

### Definition
A **Loss Function** measures how wrong the model's predictions are compared to the true labels. Lower loss means better predictions (0 = perfect).

For classification, PyTorch's **`nn.CrossEntropyLoss()`** converts the model's raw output scores (**logits**) into probabilities using **Softmax**, then heavily penalizes confident wrong predictions using the negative log of the probability assigned to the true class:
$$\mathcal{L} = -\log\left(p_{\text{true class}}\right)$$

- If the true class is `1` (Fall) and the model predicts `95%` Fall ($p=0.95$): $\text{loss} = -\log(0.95) \approx 0.051$ (tiny penalty).
- If it predicts `10%` Fall ($p=0.10$): $\text{loss} = -\log(0.10) \approx 2.30$ (large penalty).
- Random 50/50 guessing on 2 classes: $-\log(0.5) \approx 0.693$.

**Analogy:** Loss is like a golf score or a "cold/warm" error gauge—your goal during training is to push the loss as close to zero as possible.

### Why It Is Used
Accuracy ("right or wrong?") is stepped and cannot be differentiated smoothly. `CrossEntropyLoss` gives a smooth, continuous score that tells the optimizer *how much* and *in which direction* to adjust weights.

### How It Is Used in This Project
In `src/train.py`, we define `criterion = nn.CrossEntropyLoss()`. Note that `FallDetectionLSTM.forward()` returns raw **logits** `(batch, 2)` without applying Softmax, because `nn.CrossEntropyLoss()` applies `LogSoftmax` internally for numerical stability. (Softmax is only called separately in `model.predict_proba()` during inference.)

### Python Example
```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Raw logits from model for 2 samples: [score_normal, score_fall]
logits = torch.tensor([
    [3.0, 0.5],   # Confidently predicts class 0 (Normal)
    [0.2, 2.5],   # Confidently predicts class 1 (Fall)
])
true_labels = torch.tensor([0, 1], dtype=torch.long)

loss = criterion(logits, true_labels)
print(f"Good predictions loss: {loss.item():.4f}")

# Now flip the labels (confident but WRONG predictions)
wrong_labels = torch.tensor([1, 0], dtype=torch.long)
bad_loss = criterion(logits, wrong_labels)
print(f"Wrong predictions loss: {bad_loss.item():.4f}")
```

---

## 17. Backpropagation

### Definition
**Backpropagation** ("backward propagation of errors") is the algorithm used to calculate how much each weight in the neural network contributed to the loss. Starting from the final loss, it applies the calculus **chain rule** backwards, layer by layer, to compute the **gradient** ($\frac{\partial \text{Loss}}{\partial w}$) for every weight.

Simplified chain rule intuition:
If $y = f(g(w))$, then changing $w$ changes $g$, which changes $y$, which changes the Loss:
$$\frac{\partial \text{Loss}}{\partial w} = \frac{\partial \text{Loss}}{\partial y} \cdot \frac{\partial y}{\partial g} \cdot \frac{\partial g}{\partial w}$$

**Analogy:** Suppose a restaurant meal tastes too salty (high Loss). The food critic blames the final plating/sauce, who traces it back to the line cook, who traces it back to the prep chef. Everyone finds out whether they added too much or too little salt so they can fix it next time.

### Why It Is Used
Our `FallDetectionLSTM` has tens of thousands of weights across LSTM gates and linear layers. Backpropagation efficiently figures out how to tweak *every single weight simultaneously* in one backward pass.

### How It Is Used in This Project
In `src/train.py` inside `train_one_epoch()`:
```python
optimizer.zero_grad()  # 1. Clear old gradients from previous batch
loss.backward()        # 2. Backpropagation: compute gradients for all weights
optimizer.step()       # 3. Update weights using those gradients
```
During validation and inference (`validate()` and `run_inference_on_video()`), we wrap code in `with torch.no_grad():` to skip backpropagation and save memory and speed.

### Python Example
```python
import torch

# Simple equation: loss = (w * x - target)^2
x = torch.tensor(2.0)
target = torch.tensor(10.0)
w = torch.tensor(3.0, requires_grad=True)  # Weight to learn

# Forward pass: prediction = 3*2 = 6, loss = (6 - 10)^2 = 16
pred = w * x
loss = (pred - target) ** 2

# Backpropagation: d(loss)/dw = 2*(w*x - target)*x = 2*(-4)*2 = -16
loss.backward()

print(f"Loss: {loss.item()}")
print(f"Gradient dLoss/dw: {w.grad.item()}")  # Negative -> increase w to lower loss
```

---

## 18. Optimizer (Adam)

### Definition
An **Optimizer** is the algorithm that actually updates the network's weights using the gradients computed by backpropagation.

In classic Gradient Descent, weights move opposite to the gradient scaled by a **learning rate** ($\eta$):
$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Loss}}{\partial w}$$

**Adam (Adaptive Moment Estimation)** is a smarter optimizer that:
- Uses **momentum** (remembers past gradient directions so updates don't zigzag wildly).
- Adapts the **learning rate individually** for each weight based on how big its recent gradients have been.

**Analogy:** Walking downhill blindfolded in fog to reach the lowest valley (**minimum loss**):
- **Gradient** tells you which way is downhill.
- **Learning rate** is your stride length (too huge and you overshoot the valley; too tiny and it takes forever).
- **Adam** adjusts your stride smoothly depending on steepness and keeps your momentum steady.

### Why It Is Used
Plain Stochastic Gradient Descent (SGD) can get stuck or require careful manual learning-rate tuning. Adam converges much faster and more reliably out of the box—especially for RNNs/LSTMs whose gradients vary widely across parameters.

### How It Is Used in This Project
In `src/config.py` and `src/train.py`, we initialize Adam with `LEARNING_RATE = 0.001`:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
Every batch, `optimizer.step()` updates all LSTM and Linear layer weights.

### Python Example
```python
import torch

# Minimize f(w) = (w - 5)^2 starting from w = 0 using Adam
w = torch.tensor([0.0], requires_grad=True)
optimizer = torch.optim.Adam([w], lr=0.1)

for step in range(50):
    loss = (w - 5.0) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Learned w after 50 steps: {w.item():.4f} (target: 5.0)")
```

---

## 19. Overfitting

### Definition
**Overfitting** happens when a model memorizes the noise and tiny quirks of its **training data** instead of learning the general pattern. As a result, it scores near 100% on training data, but performs poorly on new, unseen people or videos.

Signs of overfitting:
- **Training loss** keeps going down.
- **Validation loss** stops improving and starts going *up*.

**Analogy:** A student who memorizes the exact word-for-word answers to a practice quiz gets 100% on the practice quiz, but fails the real exam when the questions are phrased slightly differently.

### Why It Is Used (Preventing It)
A fall detection system deployed in a real home will see *new people, different room angles, and natural movements* it never saw during training. Preventing overfitting ensures the model generalizes to real-world use.

### How It Is Used in This Project
We fight overfitting with three techniques in `src/config.py`, `src/model.py`, and `src/train.py`:
1. **Dropout** (`DROPOUT_RATE = 0.3`): Randomly turns off 30% of neurons during training.
2. **Early Stopping** (`EARLY_STOPPING_PATIENCE = 10`): Monitors validation loss after every epoch. If validation loss does not improve for 10 consecutive epochs, training stops early.
3. **Best Model Checkpointing**: Only saves `fall_detection_lstm.pth` when validation loss reaches a new low—never saving a later, overfitted version.

### Python Example
```python
# Simulating Early Stopping logic (as in src/train.py)
val_losses = [0.65, 0.50, 0.42, 0.43, 0.45, 0.48, 0.51]  # Starts overfitting after epoch 3
best_val_loss = float("inf")
patience = 3
no_improve_count = 0

for epoch, val_loss in enumerate(val_losses, start=1):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        print(f"Epoch {epoch}: val_loss={val_loss:.2f} -> New best! Saved model.")
    else:
        no_improve_count += 1
        print(f"Epoch {epoch}: val_loss={val_loss:.2f} -> No improvement ({no_improve_count}/{patience})")
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            break
```

---

## 20. Dropout

### Definition
**Dropout** is a regularization technique that randomly sets a fraction $p$ of neuron outputs to zero during each training forward pass.
- **Training mode (`model.train()`)**: Dropout is **ON**—different random neurons are disabled each batch, and remaining outputs are scaled up by $\frac{1}{1-p}$ so the expected sum stays the same.
- **Evaluation mode (`model.eval()`)**: Dropout is **OFF**—all neurons work together deterministically to make the most accurate prediction possible.

**Analogy:** Imagine a basketball team practicing where the coach randomly benches 30% of players each scrimmage. Because no player can rely on one superstar teammate doing all the work, *every* player becomes stronger and more versatile. On game day (`model.eval()`), everyone plays at full strength.

### Why It Is Used
Dropout stops neurons from co-adapting or memorizing exact training sequences, forcing the network to learn redundant, robust motion cues.

### How It Is Used in This Project
In `src/model.py`:
- Dropout (`p=0.3`) is applied between the 2 stacked LSTM layers (`dropout=0.3` inside `nn.LSTM`).
- An explicit `self.dropout = nn.Dropout(p=0.3)` layer sits between the LSTM's final hidden state and `self.fc1`.
- `src/train.py` calls `model.train()` during training epochs, and `model.eval()` in `validate()`, `evaluate.py`, and `inference.py`.

### Python Example
```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 6)

dropout.train()  # Training mode: ~50% become 0, rest scaled by 2x
print("Train pass 1:", dropout(x))
print("Train pass 2:", dropout(x))

dropout.eval()   # Eval mode: all neurons active, deterministic
print("Eval pass:   ", dropout(x))
```

---

## 21. Batch Size

### Definition
**Batch Size** is the number of training samples (sequences) fed into the model together in one forward/backward pass before updating the weights.

For example, if you have 320 training sequences and `batch_size = 32`:
- You run `320 / 32 = 10` **batches** (iterations) per epoch.

**Analogy:** Instead of grading 1 exam paper at a time and rewriting your grading rubric after each student (noisy), or waiting to read all 10,000 papers before adjusting anything (too slow and won't fit on your desk), you grade a stack of 32 papers at a time.

### Why It Is Used
1. **Memory limits:** Processing an entire large dataset at once can crash RAM/GPU memory.
2. **Speed:** GPUs and CPUs vectorize matrix math across a batch of 32 much faster than 32 separate loops of 1.
3. **Better learning:** Small batches introduce slight gradient noise that actually helps escape poor local minima.

### How It Is Used in This Project
In `src/config.py`, `BATCH_SIZE = 32`. In `src/dataset.py`, `get_data_loaders()` wraps `FallDetectionDataset` in PyTorch's `DataLoader(..., batch_size=32, shuffle=True)`, producing input tensors of shape `(32, 30, 132)` per step.

### Python Example
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 100 dummy sequences, shape (100, 30, 132), and 100 labels
X = torch.randn(100, 30, 132)
y = torch.randint(0, 2, (100,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {batch_idx + 1}: X={batch_x.shape}, y={batch_y.shape}")
```

---

## 22. Epoch

### Definition
An **Epoch** is **one complete pass** through the entire training dataset (all batches once).

- **1 Iteration / Step** = 1 batch (32 sequences) processed + 1 weight update.
- **1 Epoch** = Every training sequence seen once (`total_samples / batch_size` iterations).

**Analogy:** Reading an entire textbook from cover to cover once is **1 epoch**. Reading it chapter by chapter (batches) until you finish the book completes the epoch; reviewing the textbook 20 times (`20 epochs`) helps you master the material.

### Why It Is Used
A single pass through the data is not enough for randomly initialized weights to converge. Multiple epochs give the optimizer repeated chances to refine the decision boundary while we track progress epoch by epoch.

### How It Is Used in This Project
In `src/config.py`, `NUM_EPOCHS = 50`. Inside `src/train.py`, the outer `for epoch in range(1, num_epochs + 1):` loop runs `train_one_epoch()` followed by `validate()` after each epoch, logging training/validation loss and accuracy into `outputs/training_history.json`.

### Python Example
```python
num_samples = 100
batch_size = 32
num_epochs = 3

for epoch in range(1, num_epochs + 1):
    # Ceiling division for number of batches in 1 epoch
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Epoch {epoch}/{num_epochs} -> runs {num_batches} batch updates")
```

---

## 23. Train/Validation/Test Split

### Definition
Before training, we divide our dataset into **three separate subsets**:
1. **Training Set (~65–70%)**: Used to learn the model weights via backpropagation.
2. **Validation Set (~15–17%)**: Checked at the end of every epoch to tune hyperparameters (learning rate, architecture), trigger early stopping, and pick the best checkpoint.
3. **Test Set (~15–18%)**: Locked away until training is completely finished. Used **once** at the very end for an unbiased report of real-world performance.

**Analogy:**
- **Training set** = Homework problems you study from.
- **Validation set** = Practice mock exams to see if you're ready or need more study.
- **Test set** = The final exam you have never seen before.

### Why It Is Used
If you evaluate your model on the same data it trained on (or tune endlessly against the test set), your reported accuracy will be artificially high and misleading. Keeping three strict splits ensures honest measurement and prevents **data leakage**.

### How It Is Used in This Project
In `src/config.py` and `src/dataset.py`, for the **UP-Fall dataset** (17 human subjects), we split strictly **by Subject ID**:
- `TRAIN_SUBJECTS = Subjects 1–11` (~65%)
- `VAL_SUBJECTS = Subjects 12–14` (~17%)
- `TEST_SUBJECTS = Subjects 15–17` (~18%)

*(For synthetic pipeline testing, `prepare_data_splits` splits generated samples 70% train / 15% val / 15% test.)*

### Python Example
```python
subjects = list(range(1, 18))  # Subjects 1 to 17

train_subj = subjects[:11]     # 1..11
val_subj   = subjects[11:14]   # 12..14
test_subj  = subjects[14:]     # 15..17

print("Train subjects:", train_subj)
print("Val subjects:  ", val_subj)
print("Test subjects: ", test_subj)
# Verify zero overlap
assert set(train_subj).isdisjoint(val_subj)
assert set(train_subj).isdisjoint(test_subj)
assert set(val_subj).isdisjoint(test_subj)
```

---

## 24. Evaluation Metrics

### Definition
Because fall detection is a binary classification problem (`0 = Normal Activity`, `1 = Potential Fall`), every prediction lands in one of four **Confusion Matrix** cells:
- **True Positive (TP)**: Actually a Fall, predicted Fall (Correct alert)
- **True Negative (TN)**: Actually Normal, predicted Normal (Correct ignore)
- **False Positive (FP)**: Actually Normal, predicted Fall (**False alarm**)
- **False Negative (FN)**: Actually a Fall, predicted Normal (**Missed fall — dangerous!**)

```text
                      Predicted Normal (0)   Predicted Fall (1)
Actual Normal (0) |   True Negative (TN)   | False Positive (FP)
Actual Fall   (1) |   False Negative (FN)  | True Positive (TP)
```

From these 4 counts, we compute:
1. **Accuracy**: Overall fraction of correct predictions.
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision**: Out of all Fall alarms triggered, how many were real falls? (High precision = few false alarms.)
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity)**: Out of all *actual* falls, how many did we catch? (High recall = few missed falls.)
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: Harmonic mean of Precision and Recall (balances both).
   $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Analogy (Smoke Detector):**
- **Low Precision**: Alarm goes off every time you make toast (annoying false alarms).
- **Low Recall**: There is an actual fire, and the alarm stays silent (catastrophic failure).

### Why It Is Used
**Accuracy alone can lie on imbalanced data!** Suppose 95% of daily video is normal activity and 5% is falls. A broken model that *always* predicts "Normal" gets **95% accuracy**—yet **0% Recall** on falls! In healthcare and fall detection, **Recall is the most critical metric**, because a missed fall (`FN`) could leave someone injured on the floor without help.

### How It Is Used in This Project
In `src/evaluate.py`, `compute_metrics()` uses `scikit-learn` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) on the test set (`pos_label=1` for falls) and saves `outputs/confusion_matrix.png` and `outputs/evaluation_metrics.json`.

### Python Example
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Ground truth: 0 = Normal, 1 = Fall
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1]  # 1 false alarm (FP), 1 missed fall (FN)

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2f}")   # (4+4)/10 = 0.80
print(f"Precision: {precision_score(y_true, y_pred):.2f}")  # 4/(4+1)  = 0.80
print(f"Recall:    {recall_score(y_true, y_pred):.2f}")     # 4/(4+1)  = 0.80
print(f"F1-Score:  {f1_score(y_true, y_pred):.2f}")         # 0.80
```

---

## 25. Data Leakage

### Definition
**Data Leakage** occurs when information from outside the training set (such as validation/test samples, overlapping video windows, or test-set statistics) accidentally "leaks" into the training process. This gives the model unfair hints during training, leading to overly optimistic test scores that collapse in the real world.

**Analogy:** A student gets a sneak peek at 30% of the exact final exam questions while studying. They get an `A+` on the exam, not because they learned the subject, but because they already saw the answers.

### Why It Is Used (How We Prevent It)
In video/pose projects with **sliding windows** and **multiple subjects**, two sneaky forms of data leakage happen easily if you aren't careful:
1. **Randomly splitting sequences instead of splitting by person (subject):** Because sliding windows overlap by 50% (`stride=15`), frame `15..30` appears in both Sequence A and Sequence B! If Sequence A lands in Train and Sequence B lands in Test, the test set contains near-identical copies of training frames. Even without overlap, the model can memorize a specific person's height or posture style if that person appears in both Train and Test.
2. **Fitting normalization on the whole dataset:** Computing `MinMaxScaler` min/max values *before* splitting lets test-set min/max values influence training feature scaling.

### How It Is Used in This Project
Our codebase prevents data leakage in two strict places:
1. **Subject-wise splitting (`src/config.py` & `src/dataset.py`):** Subjects `1–11` are strictly Train, `12–14` are strictly Validation, and `15–17` are strictly Test—so the model is always tested on **completely unseen people**.
2. **Train-only scaling (`src/dataset.py` & `src/preprocessing.py`):** `MinMaxScaler` is fitted *only* on `train_features` (`fit=True`), and that saved scaler transforms `val_features` and `test_features` (`fit=False`).

### Python Example
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# BAD (Data Leakage): fitting scaler on all data before splitting
all_x = np.array([[1.0], [2.0], [3.0], [100.0]])  # 100.0 is an extreme test value
leaked_scaler = MinMaxScaler().fit(all_x)         # Train now knows test max=100!

# GOOD (No Leakage): split first, fit ONLY on train
train_x = np.array([[1.0], [2.0], [3.0]])
test_x  = np.array([[100.0]])

clean_scaler = MinMaxScaler().fit(train_x)        # Train only sees 1..3
print("Clean train scaled:\n", clean_scaler.transform(train_x))
print("Clean test scaled:\n", clean_scaler.transform(test_x))
```

---

## Quick Summary: End-to-End Flow in This Project

```text
Video / Webcam (OpenCV, BGR->RGB)
   │
   ▼
Human Pose Estimation (MediaPipe Pose -> 33 landmarks × 4 values)
   │
   ▼
Feature Vector (132 numbers per frame) + MinMaxScaler (fit on Train only)
   │
   ▼
Sliding Window (30 frames, stride 15 -> Tensor shape: (Batch, 30, 132))
   │
   ▼
FallDetectionLSTM (2-layer LSTM + Dropout 0.3 + Linear + ReLU + Linear)
   │
   ▼
Training (CrossEntropyLoss + Backprop + Adam + Early Stopping)
   │
   ▼
Evaluation on Unseen Subjects (Accuracy, Precision, Recall, F1, Confusion Matrix)
```
