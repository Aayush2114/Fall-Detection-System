"""
Training Script for Fall Detection LSTM
========================================

PURPOSE:
    Train the LSTM model on fall detection data and save the trained model.

HOW TRAINING WORKS (Simplified):
    1. Show the model a batch of sequences with known labels
    2. The model makes predictions
    3. Compare predictions with true labels using a loss function
    4. Calculate how to adjust weights to reduce the loss (backpropagation)
    5. Update weights using the optimizer (Adam)
    6. Repeat for all batches = one "epoch"
    7. Repeat for many epochs until the model learns

WHAT THIS SCRIPT DOES:
    1. Loads or generates training data
    2. Creates the LSTM model
    3. Trains for up to NUM_EPOCHS epochs
    4. Validates after each epoch to monitor for overfitting
    5. Saves the best model (lowest validation loss)
    6. Implements early stopping (stops if no improvement)
    7. Saves training history (loss/accuracy over time)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, List

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, RANDOM_SEED,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, HISTORY_SAVE_PATH,
    NUM_FEATURES, SEQUENCE_LENGTH, CLASS_LABELS,
)
from src.model import FallDetectionLSTM, save_model
from src.dataset import prepare_data_splits, get_data_loaders
from src.utils import set_seed, get_device, Timer, save_json, print_separator


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch (one pass through the training data).

    Steps for each batch:
    1. Move data to device (GPU or CPU)
    2. Forward pass: model(input) → predictions
    3. Compute loss: how far off are the predictions?
    4. Backward pass: compute gradients (how to adjust weights)
    5. Optimizer step: actually update the weights
    6. Zero gradients: reset for next batch

    Args:
        model:        The LSTM model.
        train_loader: DataLoader with training data.
        criterion:    Loss function (CrossEntropyLoss).
        optimizer:    Weight optimizer (Adam).
        device:       CPU or GPU.

    Returns:
        (average_loss, accuracy) for this epoch.
    """
    model.train()  # Enable training mode (activates dropout)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels) in enumerate(train_loader):
        # Move data to device
        sequences = sequences.to(device)  # (batch, seq_len, features)
        labels = labels.to(device)        # (batch,)

        # Forward pass
        outputs = model(sequences)        # (batch, num_classes)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights

        # Track metrics
        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on validation data (no weight updates).

    This is identical to training EXCEPT:
    - model.eval() disables dropout
    - torch.no_grad() disables gradient computation (saves memory)
    - No optimizer.step() (we don't update weights)

    Why validate?
    To check if the model generalizes to unseen data.
    If training loss decreases but validation loss increases,
    the model is OVERFITTING (memorizing training data).

    Args:
        model:      The LSTM model.
        val_loader: DataLoader with validation data.
        criterion:  Loss function.
        device:     CPU or GPU.

    Returns:
        (average_loss, accuracy) for validation data.
    """
    model.eval()  # Disable dropout, batch norm, etc.
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Don't compute gradients (save memory)
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    use_synthetic: bool = False,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> Dict:
    """
    Full training pipeline: prepare data, train model, save results.

    This is the MAIN function to call for training.

    Args:
        use_synthetic: If True, use synthetic data for testing.
        num_epochs:    Maximum number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        batch_size:    Samples per training batch.
        patience:      Early stopping patience (epochs without improvement).

    Returns:
        Dictionary with training history (losses, accuracies per epoch).
    """
    print_separator("Fall Detection Model Training")

    # ---- Setup ----
    set_seed(RANDOM_SEED)
    device = get_device()

    # ---- Data ----
    print_separator("Data Preparation")
    train_dataset, val_dataset, test_dataset = prepare_data_splits(
        use_synthetic=use_synthetic
    )
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    # ---- Model ----
    print_separator("Model Setup")
    model = FallDetectionLSTM()
    model.to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- Loss Function ----
    # CrossEntropyLoss combines LogSoftmax + NLLLoss
    # It measures how far the predicted probabilities are from the true labels
    # Perfect prediction → loss = 0; Random guessing → loss = ln(2) ≈ 0.693
    criterion = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    # Adam (Adaptive Moment Estimation) adjusts learning rate per-parameter
    # It's the most popular optimizer for deep learning
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---- Training History ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    # ---- Early Stopping State ----
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # ---- Training Loop ----
    print_separator("Training")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"Early stopping patience: {patience}\n")

    with Timer("Full training"):
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}",
                end="",
            )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model
                save_model(
                    model, MODEL_SAVE_PATH,
                    extra_info={
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                )
                print(" ← Best model saved!", end="")
            else:
                epochs_without_improvement += 1

            print()  # Newline

            # Early stopping
            if epochs_without_improvement >= patience:
                print(
                    f"\n[INFO] Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # ---- Save Training History ----
    save_json(history, HISTORY_SAVE_PATH)

    # ---- Final Summary ----
    print_separator("Training Complete")
    best_epoch = np.argmin(history["val_loss"]) + 1
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Best val accuracy: {history['val_accuracy'][best_epoch - 1]:.4f}")
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"History saved: {HISTORY_SAVE_PATH}")

    return history


# ============================================================================
# Entry point: run training from command line
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fall Detection LSTM Model")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for pipeline testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    history = train_model(
        use_synthetic=args.synthetic,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
