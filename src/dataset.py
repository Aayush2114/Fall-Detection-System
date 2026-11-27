"""
Dataset Module
==============

PURPOSE:
    Provide PyTorch Dataset classes for loading and serving training data
    to the LSTM model, plus a synthetic data generator for pipeline testing.

WHAT IS A PyTorch DATASET?
    PyTorch's DataLoader needs a Dataset object that can:
    1. Tell it how many samples exist: __len__()
    2. Return one sample by index: __getitem__(idx)

    The DataLoader then handles batching, shuffling, and parallel loading.

THIS MODULE PROVIDES:
    1. FallDetectionDataset — PyTorch Dataset for pre-processed sequences
    2. generate_synthetic_data() — Creates fake but structurally valid data
    3. prepare_data_splits() — Loads real dataset and splits by subject
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
import os

from src.config import (
    NUM_FEATURES, SEQUENCE_LENGTH, SEQUENCE_STRIDE,
    BATCH_SIZE, RANDOM_SEED, NUM_CLASSES,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS,
    FALL_ACTIVITIES, NORMAL_ACTIVITIES,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
)
from src.preprocessing import (
    normalize_features, create_labeled_sequences,
)
from src.utils import set_seed


class FallDetectionDataset(Dataset):
    """
    PyTorch Dataset for fall detection sequences.

    Each sample is:
    - Input (X):  A sequence of 30 frames, each with 132 features
                  → Tensor of shape (30, 132)
    - Label (y):  0 = Normal Activity, 1 = Potential Fall
                  → Tensor of shape () [scalar]

    Usage:
        dataset = FallDetectionDataset(sequences, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: numpy array of shape (num_samples, seq_length, num_features)
            labels:    numpy array of shape (num_samples,) with integer class labels
        """
        # Validate inputs
        assert len(sequences) == len(labels), (
            f"Mismatch: {len(sequences)} sequences but {len(labels)} labels"
        )
        assert sequences.ndim == 3, (
            f"Expected 3D sequences (samples, time, features), got shape {sequences.shape}"
        )

        # Convert to PyTorch tensors
        # float32 for input features (neural networks work with floats)
        # long (int64) for labels (CrossEntropyLoss requires integer labels)
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Count samples per class
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"[INFO] Dataset: {len(self)} samples, class distribution: {class_dist}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one sample (sequence, label) by index.

        Args:
            idx: Sample index (0 to len-1).

        Returns:
            (sequence_tensor, label_tensor)
            sequence_tensor: shape (seq_length, num_features) = (30, 132)
            label_tensor:    shape () = scalar integer
        """
        return self.sequences[idx], self.labels[idx]


def generate_synthetic_data(
    num_normal: int = 200,
    num_fall: int = 200,
    sequence_length: int = SEQUENCE_LENGTH,
    num_features: int = NUM_FEATURES,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (fake) pose sequences for testing the pipeline.

    ⚠️ WARNING: This data is NOT real. It uses simple mathematical patterns
    to simulate the difference between normal and fall movements.
    DO NOT use synthetic data results as evidence of model performance.
    It is ONLY for verifying that the code pipeline works correctly.

    How the synthetic data differs:
    - Normal: Landmarks oscillate gently around a center position
      (simulates walking, standing, sitting)
    - Fall: Landmarks show a sudden downward shift in y-coordinates
      (simulates the body dropping)

    Args:
        num_normal:      Number of normal activity sequences to generate.
        num_fall:        Number of fall sequences to generate.
        sequence_length: Frames per sequence.
        num_features:    Features per frame (132).
        seed:            Random seed for reproducibility.

    Returns:
        (sequences, labels) — ready for FallDetectionDataset
        sequences: shape (num_normal + num_fall, sequence_length, num_features)
        labels:    shape (num_normal + num_fall,)
    """
    np.random.seed(seed)
    all_sequences = []
    all_labels = []

    # --- Generate Normal Activity sequences ---
    for _ in range(num_normal):
        # Base position: random center for the person
        base = np.random.uniform(0.3, 0.7, size=(1, num_features))
        # Small oscillations over time (normal movement)
        noise = np.random.normal(0, 0.02, size=(sequence_length, num_features))
        # Gentle sinusoidal motion (walking rhythm)
        t = np.linspace(0, 2 * np.pi, sequence_length).reshape(-1, 1)
        motion = 0.01 * np.sin(t) * np.random.uniform(0.5, 1.5, size=(1, num_features))
        sequence = base + noise + motion
        sequence = np.clip(sequence, 0, 1).astype(np.float32)
        all_sequences.append(sequence)
        all_labels.append(0)  # Normal

    # --- Generate Fall sequences ---
    for _ in range(num_fall):
        base = np.random.uniform(0.3, 0.7, size=(1, num_features))
        sequence = np.zeros((sequence_length, num_features), dtype=np.float32)

        # Phase 1: Normal movement (first ~60% of sequence)
        fall_start = np.random.randint(
            int(sequence_length * 0.4), int(sequence_length * 0.7)
        )
        for t in range(fall_start):
            noise = np.random.normal(0, 0.02, size=num_features)
            sequence[t] = base.flatten() + noise

        # Phase 2: Sudden fall (remaining frames)
        # Y-coordinates (every 4th feature starting at index 1) increase rapidly
        # (in MediaPipe, higher y = lower in the frame = closer to ground)
        fall_speed = np.random.uniform(0.03, 0.08)
        for t in range(fall_start, sequence_length):
            dt = t - fall_start
            noise = np.random.normal(0, 0.03, size=num_features)
            drop = np.zeros(num_features)
            # Increase y-coordinates (indices 1, 5, 9, 13, ... every 4th)
            drop[1::4] = fall_speed * dt  # y values increase (falling down)
            drop[2::4] = -0.01 * dt       # z values decrease (depth change)
            sequence[t] = base.flatten() + noise + drop

        sequence = np.clip(sequence, 0, 1).astype(np.float32)
        all_sequences.append(sequence)
        all_labels.append(1)  # Fall

    sequences = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle the data
    indices = np.random.permutation(len(labels))
    sequences = sequences[indices]
    labels = labels[indices]

    print(f"[INFO] Generated synthetic data:")
    print(f"  Normal: {num_normal}, Fall: {num_fall}, Total: {len(labels)}")
    print(f"  Shape: sequences={sequences.shape}, labels={labels.shape}")

    return sequences, labels


def load_upfall_skeleton_data(
    data_dir: Path = RAW_DATA_DIR,
) -> Optional[Dict]:
    """
    Load the UP-Fall 3D Skeleton dataset from the data/raw/ directory.

    Expected directory structure after download:
        data/raw/
        ├── Subject1/
        │   ├── Activity1/
        │   │   ├── Trial1.csv
        │   │   ├── Trial2.csv
        │   │   └── Trial3.csv
        │   ├── Activity2/
        │   ...
        ├── Subject2/
        ...

    Each CSV file contains frame-by-frame landmark data.

    If the data is not found, returns None and prints download instructions.

    Args:
        data_dir: Path to the raw data directory.

    Returns:
        Dictionary with 'features', 'labels', 'subjects', 'activities'
        or None if data not found.
    """
    data_dir = Path(data_dir)

    # Check for CSV files in the raw directory (flexible structure)
    csv_files = list(data_dir.rglob("*.csv"))

    if not csv_files:
        print("[INFO] UP-Fall skeleton data not found in data/raw/")
        print("[INFO] To use real data, follow the instructions in data/README.md")
        print("[INFO] Using synthetic data for pipeline testing instead.")
        return None

    print(f"[INFO] Found {len(csv_files)} CSV files in {data_dir}")

    import pandas as pd

    all_features = []
    all_labels = []
    all_subjects = []
    all_activities = []

    for csv_file in sorted(csv_files):
        try:
            # Try to parse subject and activity from path
            parts = csv_file.relative_to(data_dir).parts
            # Attempt to extract subject/activity numbers from directory names
            subject_id = None
            activity_id = None

            for part in parts:
                part_lower = part.lower()
                if "subject" in part_lower:
                    nums = [int(c) for c in part if c.isdigit()]
                    if nums:
                        subject_id = int("".join(str(n) for n in nums))
                elif "activity" in part_lower:
                    nums = [int(c) for c in part if c.isdigit()]
                    if nums:
                        activity_id = int("".join(str(n) for n in nums))

            if subject_id is None or activity_id is None:
                continue

            # Read the CSV
            df = pd.read_csv(csv_file)

            # Expect at least NUM_FEATURES columns of landmark data
            if df.shape[1] < NUM_FEATURES:
                continue

            # Take the first NUM_FEATURES columns as landmark features
            features = df.iloc[:, :NUM_FEATURES].values.astype(np.float32)

            # Determine label: falls (activities 1-5) vs normal (6-11)
            label = 1 if activity_id in FALL_ACTIVITIES else 0

            all_features.append(features)
            all_labels.append(label)
            all_subjects.append(subject_id)
            all_activities.append(activity_id)

        except Exception as e:
            print(f"[WARNING] Could not parse {csv_file}: {e}")
            continue

    if not all_features:
        print("[WARNING] No valid data files could be parsed.")
        return None

    print(f"[INFO] Loaded {len(all_features)} recordings from {len(set(all_subjects))} subjects")

    return {
        "features": all_features,     # List of arrays, each (num_frames, 132)
        "labels": all_labels,          # List of int labels (0 or 1)
        "subjects": all_subjects,      # List of subject IDs
        "activities": all_activities,  # List of activity IDs
    }


def prepare_data_splits(
    use_synthetic: bool = False,
    synthetic_samples: int = 200,
) -> Tuple[FallDetectionDataset, FallDetectionDataset, FallDetectionDataset]:
    """
    Prepare train, validation, and test datasets.

    Strategy:
    1. Try to load real UP-Fall data from data/raw/
    2. If not available, fall back to synthetic data
    3. Split by subject ID (prevents data leakage)
    4. Normalize features (fit on train, transform val/test)
    5. Create sliding window sequences
    6. Return PyTorch Dataset objects

    Args:
        use_synthetic:    Force synthetic data even if real data exists.
        synthetic_samples: Number of samples per class for synthetic data.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    set_seed(RANDOM_SEED)

    if use_synthetic:
        real_data = None
    else:
        real_data = load_upfall_skeleton_data()

    if real_data is None:
        # === Synthetic Data Path ===
        print("\n[INFO] Using SYNTHETIC data (for pipeline testing only)")
        print("[WARNING] Results with synthetic data do NOT represent real performance!\n")

        sequences, labels = generate_synthetic_data(
            num_normal=synthetic_samples,
            num_fall=synthetic_samples,
        )

        # Split: 70% train, 15% val, 15% test
        n = len(labels)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_seq, train_lab = sequences[:n_train], labels[:n_train]
        val_seq, val_lab = sequences[n_train:n_train + n_val], labels[n_train:n_train + n_val]
        test_seq, test_lab = sequences[n_train + n_val:], labels[n_train + n_val:]

    else:
        # === Real Data Path ===
        print("\n[INFO] Using REAL UP-Fall skeleton data")

        # Split recordings by subject ID
        train_features, train_labels_list = [], []
        val_features, val_labels_list = [], []
        test_features, test_labels_list = [], []

        for feat, lab, subj in zip(
            real_data["features"], real_data["labels"], real_data["subjects"]
        ):
            if subj in TRAIN_SUBJECTS:
                train_features.append(feat)
                train_labels_list.append(lab)
            elif subj in VAL_SUBJECTS:
                val_features.append(feat)
                val_labels_list.append(lab)
            elif subj in TEST_SUBJECTS:
                test_features.append(feat)
                test_labels_list.append(lab)

        # Normalize: fit on training data only
        if train_features:
            all_train = np.vstack(train_features)
            all_train_norm, scaler = normalize_features(
                all_train, fit=True, save_path=PROCESSED_DATA_DIR / "scaler.pkl"
            )
        else:
            print("[ERROR] No training data found!")
            return prepare_data_splits(use_synthetic=True)

        # Create sequences for each split
        def make_sequences(features_list, labels_list, scaler, fit=False):
            all_seq, all_lab = [], []
            for feat, lab in zip(features_list, labels_list):
                feat_norm, _ = normalize_features(feat, scaler=scaler, fit=False)
                seqs, labs = create_labeled_sequences(feat_norm, lab)
                all_seq.append(seqs)
                all_lab.append(labs)
            if all_seq:
                return np.concatenate(all_seq), np.concatenate(all_lab)
            return np.empty((0, SEQUENCE_LENGTH, NUM_FEATURES)), np.empty(0, dtype=np.int64)

        train_seq, train_lab = make_sequences(train_features, train_labels_list, scaler)
        val_seq, val_lab = make_sequences(val_features, val_labels_list, scaler)
        test_seq, test_lab = make_sequences(test_features, test_labels_list, scaler)

    # Create PyTorch datasets
    print(f"\n[INFO] Data splits:")
    print(f"  Train: {len(train_lab)} sequences")
    print(f"  Val:   {len(val_lab)} sequences")
    print(f"  Test:  {len(test_lab)} sequences")

    train_dataset = FallDetectionDataset(train_seq, train_lab)
    val_dataset = FallDetectionDataset(val_seq, val_lab)
    test_dataset = FallDetectionDataset(test_seq, test_lab)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(
    train_dataset: FallDetectionDataset,
    val_dataset: FallDetectionDataset,
    test_dataset: FallDetectionDataset,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.

    DataLoaders handle:
    - Batching: Group samples into batches of `batch_size`
    - Shuffling: Randomize order each epoch (training only)
    - Parallel loading: Use multiple CPU threads (num_workers)

    Args:
        train_dataset, val_dataset, test_dataset: PyTorch Datasets.
        batch_size: Number of samples per batch.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Shuffle training data each epoch
        drop_last=False,
        num_workers=0,       # 0 = load in main process (most compatible)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle validation/test
        drop_last=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print(f"[INFO] DataLoaders created (batch_size={batch_size})")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ============================================================================
# Direct execution test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Module — Test")
    print("=" * 60)

    # Test synthetic data generation
    print("\n--- Synthetic Data Test ---")
    sequences, labels = generate_synthetic_data(num_normal=50, num_fall=50)
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label values: {np.unique(labels, return_counts=True)}")

    # Test Dataset class
    print("\n--- FallDetectionDataset Test ---")
    dataset = FallDetectionDataset(sequences, labels)
    sample_x, sample_y = dataset[0]
    print(f"  Sample input shape: {sample_x.shape}")
    print(f"  Sample label: {sample_y.item()}")
    print(f"  Input dtype: {sample_x.dtype}, Label dtype: {sample_y.dtype}")

    # Test DataLoader
    print("\n--- DataLoader Test ---")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"  Batch input shape: {batch_x.shape}")
    print(f"  Batch labels shape: {batch_y.shape}")
    print(f"  Expected: (16, {SEQUENCE_LENGTH}, {NUM_FEATURES})")

    print("\n[OK] All dataset tests passed!")
