"""
Configuration File for Fall Detection System
=============================================

This file centralizes ALL configurable parameters for the project.
Instead of hardcoding values in multiple files, we define them here once.

This makes it easy to:
- Change hyperparameters (learning rate, batch size, etc.)
- Switch file paths
- Adjust model architecture
- Modify visualization settings

Every other module imports from this file.
"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

# Root directory of the project (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model save directory
MODELS_DIR = PROJECT_ROOT / "models"

# Output directory (for saved videos, plots, etc.)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MediaPipe Pose Configuration
# ============================================================================

# MediaPipe Pose detects 33 body landmarks (joints) on a person.
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
NUM_LANDMARKS = 33

# Each landmark has 4 values:
#   x - horizontal position (0.0 = left edge, 1.0 = right edge of frame)
#   y - vertical position (0.0 = top, 1.0 = bottom of frame)
#   z - depth relative to hips (smaller = closer to camera)
#   visibility - confidence that this joint is visible (0.0 to 1.0)
LANDMARK_FEATURES = 4

# Total features per frame = 33 landmarks × 4 values = 132
NUM_FEATURES = NUM_LANDMARKS * LANDMARK_FEATURES

# MediaPipe confidence thresholds
# Detection: how confident the model must be to initially find a person
# Tracking: how confident to keep tracking a person between frames
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================================
# Sequence / Sliding Window Configuration
# ============================================================================

# Number of consecutive frames grouped into one sequence.
# At 30 FPS, 30 frames = 1 second of movement.
# Falls typically take 0.5-2 seconds, so 1 second captures the dynamics.
SEQUENCE_LENGTH = 30

# How many frames to skip between consecutive sequences.
# stride=15 means each sequence overlaps 50% with the next one.
# This creates more training samples from limited data.
SEQUENCE_STRIDE = 15

# ============================================================================
# Model Hyperparameters
# ============================================================================

# LSTM (Long Short-Term Memory) network architecture:
# Two stacked LSTM layers process the temporal sequence of pose data.
LSTM_HIDDEN_SIZE_1 = 128   # Hidden units in first LSTM layer
LSTM_HIDDEN_SIZE_2 = 64    # Hidden units in second LSTM layer
LSTM_NUM_LAYERS = 2        # Number of stacked LSTM layers
DROPOUT_RATE = 0.3         # Probability of dropping neurons during training
                           # (prevents overfitting = memorizing training data)

# Fully connected (FC) classification layers after LSTM
FC_HIDDEN_SIZE = 32        # Hidden units in the FC layer

# Output classes
NUM_CLASSES = 2            # Binary: Normal Activity vs Potential Fall

# Human-readable class names
CLASS_LABELS = {
    0: "Normal Activity",
    1: "Potential Fall"
}

# ============================================================================
# Training Hyperparameters
# ============================================================================

BATCH_SIZE = 32            # Number of sequences processed together
LEARNING_RATE = 0.001      # Step size for weight updates (Adam optimizer)
NUM_EPOCHS = 50            # Maximum training iterations over full dataset
EARLY_STOPPING_PATIENCE = 10  # Stop if validation loss doesn't improve
                               # for this many consecutive epochs

# Reproducibility: fixing this seed makes results repeatable
RANDOM_SEED = 42

# ============================================================================
# Dataset Split Configuration (by Subject ID)
# ============================================================================

# UP-Fall dataset has 17 subjects performing activities.
# We split BY SUBJECT to prevent data leakage:
# - If the same person appears in both training and test sets,
#   the model might just learn to recognize that person,
#   not the activity pattern.

TRAIN_SUBJECTS = list(range(1, 12))    # Subjects 1-11 (65%)
VAL_SUBJECTS = list(range(12, 15))     # Subjects 12-14 (17%)
TEST_SUBJECTS = list(range(15, 18))    # Subjects 15-17 (18%)

# Activity labels from UP-Fall dataset
# Activities 1-5 are falls, Activities 6-11 are normal daily activities
FALL_ACTIVITIES = [1, 2, 3, 4, 5]
NORMAL_ACTIVITIES = [6, 7, 8, 9, 10, 11]

# ============================================================================
# Visualization Settings
# ============================================================================

# Colors in BGR format (OpenCV uses BGR, not RGB)
NORMAL_COLOR = (0, 255, 0)       # Green - safe, normal activity
FALL_COLOR = (0, 0, 255)         # Red - warning, potential fall
SKELETON_COLOR = (255, 255, 0)   # Cyan - skeleton lines
LANDMARK_COLOR = (0, 255, 255)   # Yellow - landmark points

# Font settings for text overlay on video
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# ============================================================================
# File Paths for Saved Artifacts
# ============================================================================

MODEL_SAVE_PATH = MODELS_DIR / "fall_detection_lstm.pth"
SCALER_SAVE_PATH = MODELS_DIR / "feature_scaler.pkl"
HISTORY_SAVE_PATH = OUTPUTS_DIR / "training_history.json"
