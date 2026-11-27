"""
Utility Functions for Fall Detection System
============================================

This module provides helper functions used by multiple other modules:
- Random seed setting (for reproducible experiments)
- Device detection (CPU vs GPU)
- Timing utilities
- File I/O helpers

Think of this as a "toolbox" that other files reach into when needed.
"""

import os
import random
import time
import json
import numpy as np
import torch
from pathlib import Path

from src.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Why is this needed?
    Neural networks use random numbers for:
    - Initial weight values
    - Shuffling training data
    - Dropout (randomly disabling neurons)

    Without a fixed seed, you get slightly different results each run.
    Setting a seed means: same seed → same random numbers → same results.

    Args:
        seed: Integer seed value. Default from config.py (42).
    """
    random.seed(seed)                     # Python's built-in random
    np.random.seed(seed)                  # NumPy random
    torch.manual_seed(seed)               # PyTorch CPU random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # PyTorch GPU random
        torch.cuda.manual_seed_all(seed)  # All GPUs (if multiple)
    # Make CUDA operations deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    PyTorch can run computations on:
    - CPU: Available everywhere, slower for neural networks
    - GPU (CUDA): Much faster for matrix operations (10-100x speedup)

    This function automatically picks GPU if available.

    Returns:
        torch.device object — either 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available. Using CPU.")
    return device


class Timer:
    """
    Simple timer for measuring how long operations take.

    Usage:
        with Timer("Training"):
            train_model()
        # Prints: "[TIMER] Training completed in 45.23 seconds."
    """

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            print(f"[TIMER] {self.description} completed in {elapsed:.2f} seconds.")
        else:
            minutes = elapsed / 60
            print(f"[TIMER] {self.description} completed in {minutes:.1f} minutes.")


def save_json(data: dict, filepath) -> None:
    """
    Save a Python dictionary to a JSON file.

    Args:
        data: Dictionary to save.
        filepath: Where to save (string or Path).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved: {filepath}")


def load_json(filepath) -> dict:
    """
    Load a Python dictionary from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dictionary with the loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def print_separator(title: str = "", char: str = "=", length: int = 60) -> None:
    """Print a visual separator line for clean console output."""
    if title:
        padding = (length - len(title) - 2) // 2
        line = f"{char * padding} {title} {char * padding}"
        print(f"\n{line}")
    else:
        print(char * length)
