"""
Preprocessing Module
====================

PURPOSE:
    Transform raw pose landmarks into normalized, fixed-length sequences
    ready for the LSTM model.

PIPELINE:
    Raw landmarks (N, 132)
        → Normalize features to [0, 1] range
        → Build sliding window sequences of 30 frames each
        → Label each sequence (fall or normal)
        → Output tensors ready for PyTorch

WHY NORMALIZE?
    Raw landmark values have different scales:
    - x, y: 0.0 to 1.0 (already normalized by MediaPipe)
    - z: can be negative or positive (depth)
    - visibility: 0.0 to 1.0

    Normalizing all features to the same range helps the neural network
    learn faster and more stably (avoids one feature dominating others).

WHY SLIDING WINDOWS?
    A single frame can't tell us if someone is falling — it's just a pose.
    We need to see how the pose CHANGES over time.
    A sliding window groups consecutive frames into a sequence:

    Frame: [1] [2] [3] [4] [5] [6] [7] [8] [9] ...
    Seq 1: [=========30 frames=========]
    Seq 2:          [=========30 frames=========]
                    ↑ stride=15 (50% overlap)

    Each sequence becomes one input sample for the LSTM.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    NUM_FEATURES, SEQUENCE_LENGTH, SEQUENCE_STRIDE,
    SCALER_SAVE_PATH,
)


def normalize_features(
    features: np.ndarray,
    scaler: Optional[MinMaxScaler] = None,
    fit: bool = True,
    save_path: Optional[Path] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize feature values to the [0, 1] range using Min-Max scaling.

    Min-Max scaling formula for each feature:
        x_normalized = (x - x_min) / (x_max - x_min)

    IMPORTANT: We fit the scaler on TRAINING data only, then use the same
    scaler to transform validation and test data. This prevents data leakage
    (the model shouldn't "see" test data statistics during training).

    Args:
        features:  Array of shape (num_frames, 132) — raw landmark values.
        scaler:    Pre-fitted scaler (use for val/test data). None to create new.
        fit:       If True, fit the scaler on this data (training mode).
                   If False, only transform using existing scaler.
        save_path: Where to save the fitted scaler (pickle file).

    Returns:
        (normalized_features, scaler)
        normalized_features: Same shape as input, values in [0, 1].
        scaler:             The fitted MinMaxScaler object.
    """
    if features.shape[0] == 0:
        print("[WARNING] Empty features array — nothing to normalize.")
        if scaler is None:
            scaler = MinMaxScaler()
        return features, scaler

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if fit:
        # fit_transform: learn min/max from this data AND transform it
        normalized = scaler.fit_transform(features)
        print(f"[INFO] Scaler fitted on {features.shape[0]} frames.")

        # Save the scaler so we can reuse it during inference
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"[INFO] Scaler saved: {save_path}")
    else:
        # transform only: use previously learned min/max values
        normalized = scaler.transform(features)

    return normalized.astype(np.float32), scaler


def load_scaler(scaler_path: Path = SCALER_SAVE_PATH) -> MinMaxScaler:
    """
    Load a previously saved feature scaler from disk.

    This is used during inference to apply the SAME normalization
    that was used during training.

    Args:
        scaler_path: Path to the saved .pkl file.

    Returns:
        Fitted MinMaxScaler object.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}\n"
            f"You need to train the model first (which saves the scaler)."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[INFO] Scaler loaded: {scaler_path}")
    return scaler


def create_sequences(
    features: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = SEQUENCE_STRIDE,
) -> np.ndarray:
    """
    Create fixed-length sequences from frame-level features using a sliding window.

    Example with sequence_length=4, stride=2:
        Input frames:  [f0, f1, f2, f3, f4, f5, f6, f7]
        Sequence 0:    [f0, f1, f2, f3]
        Sequence 1:    [f2, f3, f4, f5]
        Sequence 2:    [f4, f5, f6, f7]

    Args:
        features:        Array of shape (num_frames, num_features).
        sequence_length: Number of consecutive frames per sequence.
        stride:          Step size between consecutive sequences.

    Returns:
        Array of shape (num_sequences, sequence_length, num_features).
    """
    num_frames, num_features = features.shape

    if num_frames < sequence_length:
        print(
            f"[WARNING] Only {num_frames} frames available, but need "
            f"{sequence_length}. Padding with zeros."
        )
        # Pad with zeros if not enough frames
        padded = np.zeros((sequence_length, num_features), dtype=np.float32)
        padded[:num_frames] = features
        return padded[np.newaxis, ...]  # shape: (1, seq_len, features)

    sequences = []
    for start in range(0, num_frames - sequence_length + 1, stride):
        end = start + sequence_length
        seq = features[start:end]
        sequences.append(seq)

    sequences_array = np.array(sequences, dtype=np.float32)
    print(
        f"[INFO] Created {len(sequences)} sequences "
        f"(length={sequence_length}, stride={stride}) "
        f"from {num_frames} frames."
    )
    return sequences_array


def create_labeled_sequences(
    features: np.ndarray,
    label: int,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = SEQUENCE_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences AND assign a label to each one.

    This is used during training data preparation:
    - All sequences from a "fall" video get label=1
    - All sequences from a "normal" video get label=0

    Args:
        features:        Frame-level features, shape (num_frames, 132).
        label:           Integer class label (0=normal, 1=fall).
        sequence_length: Frames per sequence.
        stride:          Step between sequences.

    Returns:
        (sequences, labels)
        sequences: shape (num_sequences, sequence_length, 132)
        labels:    shape (num_sequences,) — all set to `label`
    """
    sequences = create_sequences(features, sequence_length, stride)
    labels = np.full(len(sequences), label, dtype=np.int64)
    return sequences, labels


def interpolate_missing_frames(
    landmarks: np.ndarray,
    valid_indices: list,
    total_frames: int,
) -> np.ndarray:
    """
    Fill in missing frames where no pose was detected using linear interpolation.

    Sometimes MediaPipe can't detect a person in every frame (occlusion,
    fast motion, etc.). This function fills gaps by interpolating between
    the nearest valid frames.

    Args:
        landmarks:     Array of detected landmarks, shape (N, 132).
        valid_indices: Frame numbers where detection succeeded.
        total_frames:  Total number of frames in the video.

    Returns:
        Array of shape (total_frames, 132) with interpolated values.
    """
    if len(valid_indices) == 0:
        print("[WARNING] No valid frames to interpolate from.")
        return np.zeros((total_frames, landmarks.shape[1]), dtype=np.float32)

    num_features = landmarks.shape[1]
    full_landmarks = np.zeros((total_frames, num_features), dtype=np.float32)

    # Place known values
    for i, idx in enumerate(valid_indices):
        if idx < total_frames:
            full_landmarks[idx] = landmarks[i]

    # Linear interpolation for each feature column
    valid_indices_arr = np.array(valid_indices)
    for feat in range(num_features):
        known_values = landmarks[:, feat]
        full_landmarks[:, feat] = np.interp(
            x=np.arange(total_frames),
            xp=valid_indices_arr,
            fp=known_values,
        )

    filled_count = total_frames - len(valid_indices)
    if filled_count > 0:
        print(f"[INFO] Interpolated {filled_count} missing frames.")

    return full_landmarks


# ============================================================================
# Direct execution test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Preprocessing Module — Test")
    print("=" * 60)

    # Create dummy data to test the pipeline
    print("\nCreating dummy data (100 frames, 132 features)...")
    dummy_features = np.random.rand(100, NUM_FEATURES).astype(np.float32)

    # Test normalization
    print("\n--- Normalization Test ---")
    normalized, scaler = normalize_features(dummy_features, fit=True)
    print(f"  Input range:  [{dummy_features.min():.4f}, {dummy_features.max():.4f}]")
    print(f"  Output range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    # Test sequence creation
    print("\n--- Sequence Creation Test ---")
    sequences = create_sequences(normalized, sequence_length=30, stride=15)
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Expected: (num_seq, 30, 132)")

    # Test labeled sequences
    print("\n--- Labeled Sequences Test ---")
    seqs, labels = create_labeled_sequences(normalized, label=1)
    print(f"  Sequences: {seqs.shape}, Labels: {labels.shape}")
    print(f"  All labels are 1: {np.all(labels == 1)}")

    print("\n[OK] All preprocessing tests passed!")
