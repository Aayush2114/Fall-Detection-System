"""
Model Evaluation Module
========================

PURPOSE:
    Evaluate the trained LSTM model and generate performance reports.

METRICS EXPLAINED:
    - Accuracy: What fraction of ALL predictions were correct?
      Simple but can be misleading with imbalanced data.

    - Precision: Of all samples predicted as "Fall", how many actually were?
      High precision = few false alarms.

    - Recall: Of all ACTUAL falls, how many did we detect?
      High recall = we miss very few real falls.
      ⚠️ For fall detection, recall is CRITICAL — a missed fall could
      mean an elderly person lying on the floor without help.

    - F1-Score: Harmonic mean of precision and recall.
      Balances both concerns.

    - Confusion Matrix: Shows the full breakdown:
      True Positives (TP):  Correctly detected falls
      True Negatives (TN):  Correctly identified normal activity
      False Positives (FP): False alarms (predicted fall, was normal)
      False Negatives (FN): Missed falls (predicted normal, was actually fall)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no window needed)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
from typing import Dict, Tuple, List

from src.config import (
    CLASS_LABELS, NUM_CLASSES, MODEL_SAVE_PATH,
    HISTORY_SAVE_PATH, OUTPUTS_DIR,
)
from src.model import FallDetectionLSTM, load_model
from src.dataset import prepare_data_splits, get_data_loaders
from src.utils import get_device, load_json, print_separator


def get_predictions(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model on all data in the loader and collect predictions.

    Args:
        model:       Trained model in eval mode.
        data_loader: DataLoader with test/validation data.
        device:      CPU or GPU.

    Returns:
        (true_labels, predicted_labels, predicted_probabilities)
        All as numpy arrays.
    """
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_true.extend(labels.numpy())
            all_pred.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1),
        "f1_score": f1_score(y_true, y_pred, average="binary", pos_label=1),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = None,
) -> None:
    """
    Generate and save a confusion matrix visualization.

    The confusion matrix is a 2×2 table showing:
                        Predicted Normal  |  Predicted Fall
    Actual Normal:      True Negative (TN) |  False Positive (FP)
    Actual Fall:        False Negative (FN) |  True Positive (TP)

    Args:
        y_true:    Ground truth labels.
        y_pred:    Predicted labels.
        save_path: Where to save the plot image.
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    # Add color bar
    plt.colorbar(im)

    # Add labels
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16)

    # Add numbers in each cell
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved: {save_path}")
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Plot training and validation loss/accuracy curves.

    These plots help diagnose training:
    - Both curves decreasing: Good learning!
    - Train decreasing, val increasing: Overfitting
    - Both flat: Learning rate too low or model too simple

    Args:
        history:  Dictionary with 'train_loss', 'val_loss', etc.
        save_dir: Directory to save plot images.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-o", label="Training Loss", markersize=3)
    ax.plot(epochs, history["val_loss"], "r-o", label="Validation Loss", markersize=3)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Training & Validation Loss", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = save_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Loss plot saved: {loss_path}")
    plt.close()

    # ---- Accuracy Plot ----
    if "train_accuracy" in history and history["train_accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history["train_accuracy"], "b-o",
                label="Training Accuracy", markersize=3)
        ax.plot(epochs, history["val_accuracy"], "r-o",
                label="Validation Accuracy", markersize=3)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title("Training & Validation Accuracy", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        acc_path = save_dir / "accuracy_curves.png"
        plt.savefig(acc_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Accuracy plot saved: {acc_path}")
        plt.close()


def evaluate_model(
    model_path: Path = MODEL_SAVE_PATH,
    use_synthetic: bool = False,
) -> Dict:
    """
    Full evaluation pipeline: load model, run on test data, generate reports.

    Args:
        model_path:    Path to the saved model .pth file.
        use_synthetic: Use synthetic data for testing.

    Returns:
        Dictionary with all metrics and paths to generated plots.
    """
    print_separator("Model Evaluation")

    device = get_device()

    # Load model
    model = load_model(model_path, device)

    # Prepare test data
    _, _, test_dataset = prepare_data_splits(use_synthetic=use_synthetic)
    _, _, test_loader = get_data_loaders(
        test_dataset, test_dataset, test_dataset  # Only need test loader
    )

    # Get predictions
    print("\n[INFO] Running predictions on test data...")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # Print results
    print_separator("Evaluation Results")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    print(f"\n  Classification Report:")
    class_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Why recall matters for fall detection:
    print("  ⚠️  Note on Recall:")
    print("  In fall detection, RECALL is the most critical metric.")
    print("  A False Negative (missed fall) means a person who fell")
    print("  does not receive help. A False Positive (false alarm) is")
    print("  inconvenient but not dangerous.")
    print("  Aim for high recall, even at the cost of some precision.\n")

    # Generate plots
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    # Plot training history if available
    if HISTORY_SAVE_PATH.exists():
        history = load_json(HISTORY_SAVE_PATH)
        plot_training_history(history)

    # Save metrics to JSON
    metrics_path = OUTPUTS_DIR / "evaluation_metrics.json"
    from src.utils import save_json
    save_json(metrics, metrics_path)

    return metrics


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Fall Detection Model")
    parser.add_argument(
        "--model", type=str, default=str(MODEL_SAVE_PATH),
        help="Path to trained model file"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic test data"
    )
    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=Path(args.model),
        use_synthetic=args.synthetic,
    )
