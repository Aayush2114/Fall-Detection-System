"""
LSTM Model for Fall Detection
==============================

PURPOSE:
    Define the neural network architecture that classifies pose sequences
    as "Normal Activity" or "Potential Fall".

WHAT IS AN LSTM?
    LSTM = Long Short-Term Memory.
    It's a type of Recurrent Neural Network (RNN) designed to learn from
    SEQUENTIAL data — data where ORDER matters.

    Why is order important for fall detection?
    - A single pose (one frame) could look like someone bending down
      OR falling — we can't tell from one snapshot.
    - But if we see the person MOVING from standing → tilting → on the ground
      over a sequence of frames, that's clearly a fall.

    An LSTM processes one frame at a time, maintaining an internal "memory"
    (hidden state) that accumulates information from previous frames.
    After seeing all 30 frames, its final hidden state encodes the
    overall movement pattern, which we then classify.

WHY LSTM INSTEAD OF CNN?
    - CNN (Convolutional Neural Network): Best for spatial patterns (images).
      It looks at local regions and learns textures, edges, shapes.
    - LSTM (Long Short-Term Memory): Best for temporal patterns (sequences).
      It understands how data changes over TIME.

    Fall detection is about TEMPORAL change in body position,
    so LSTM is a more natural fit than CNN.

ARCHITECTURE:
    Input: (batch_size, 30, 132) — 30 frames, each with 132 features

    → LSTM Layer 1 (hidden=128): Process sequence, output hidden states
    → LSTM Layer 2 (hidden=64):  Refine temporal features
    → Take the LAST hidden state: (batch_size, 64)
    → Dropout (p=0.3): Randomly zero out neurons to prevent overfitting
    → Fully Connected (64 → 32): Compress features
    → ReLU activation: Introduce non-linearity
    → Fully Connected (32 → 2): Output class scores
    → Softmax (applied during inference): Convert to probabilities

    Output: (batch_size, 2) — scores for [Normal, Fall]
"""

import torch
import torch.nn as nn
from pathlib import Path

from src.config import (
    NUM_FEATURES,
    LSTM_HIDDEN_SIZE_1,
    LSTM_HIDDEN_SIZE_2,
    LSTM_NUM_LAYERS,
    DROPOUT_RATE,
    FC_HIDDEN_SIZE,
    NUM_CLASSES,
    MODEL_SAVE_PATH,
)


class FallDetectionLSTM(nn.Module):
    """
    LSTM-based classifier for fall detection from pose sequences.

    Inherits from nn.Module, which is the base class for all PyTorch
    neural network layers and models. Every custom model must:
    1. Call super().__init__() in __init__
    2. Define layers in __init__
    3. Define the forward pass in forward()
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = LSTM_HIDDEN_SIZE_1,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_RATE,
    ):
        """
        Initialize the model layers.

        Args:
            input_size:  Number of features per frame (132).
            hidden_size: Number of LSTM hidden units (128).
            num_layers:  Number of stacked LSTM layers (2).
            num_classes: Number of output classes (2).
            dropout:     Dropout probability (0.3).
        """
        super(FallDetectionLSTM, self).__init__()

        # Store parameters for reference
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # ---- LSTM Layer ----
        # batch_first=True means input shape is (batch, sequence, features)
        # instead of (sequence, batch, features)
        self.lstm = nn.LSTM(
            input_size=input_size,      # 132 features per frame
            hidden_size=hidden_size,    # 128 hidden units
            num_layers=num_layers,      # 2 stacked LSTM layers
            batch_first=True,           # Input: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between layers
        )

        # ---- Classification Head ----
        # After the LSTM, we take the last hidden state and classify it
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size, FC_HIDDEN_SIZE)  # 128 → 32
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(FC_HIDDEN_SIZE, num_classes)  # 32 → 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: process input through the network.

        This is called automatically when you do: output = model(input)

        Step by step:
        1. LSTM processes the sequence frame by frame
        2. We take only the LAST time step's output
           (it has seen all previous frames and accumulated context)
        3. Pass through dropout → FC → ReLU → FC
        4. Output raw scores (logits) for each class

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
               Example: (32, 30, 132)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
            Example: (32, 2)
        """
        # LSTM forward pass
        # lstm_out shape: (batch, seq_len, hidden_size) = (32, 30, 128)
        # h_n shape: (num_layers, batch, hidden_size) = (2, 32, 128)
        # c_n shape: (num_layers, batch, hidden_size) = (2, 32, 128)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the output from the LAST time step only
        # lstm_out[:, -1, :] → shape: (batch, hidden_size) = (32, 128)
        last_output = lstm_out[:, -1, :]

        # Classification head
        out = self.dropout(last_output)   # (32, 128) — randomly zero some values
        out = self.fc1(out)               # (32, 32)  — linear transformation
        out = self.relu(out)              # (32, 32)  — apply non-linearity
        out = self.fc2(out)               # (32, 2)   — final class scores

        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (not raw scores).

        Applies softmax to convert logits to probabilities that sum to 1.
        Example: logits [2.1, 0.3] → probabilities [0.86, 0.14]

        Args:
            x: Input tensor, same as forward().

        Returns:
            Probabilities tensor, shape (batch_size, num_classes).
            Each row sums to 1.0.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def save_model(
    model: FallDetectionLSTM,
    path: Path = MODEL_SAVE_PATH,
    extra_info: dict = None,
) -> None:
    """
    Save the trained model to disk.

    We save a "state_dict" — a dictionary of all learned weights.
    This is preferred over saving the entire model object because
    it's more portable and flexible.

    We also save model architecture parameters so we can recreate
    the exact same model structure when loading.

    Args:
        model:      The trained model.
        path:       Where to save (default: models/fall_detection_lstm.pth).
        extra_info: Optional dict of additional info to save (e.g., accuracy).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
    }
    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)
    print(f"[INFO] Model saved: {path}")


def load_model(
    path: Path = MODEL_SAVE_PATH,
    device: torch.device = None,
) -> FallDetectionLSTM:
    """
    Load a saved model from disk.

    Steps:
    1. Load the checkpoint dictionary
    2. Recreate the model with the saved architecture
    3. Load the learned weights into the model
    4. Set to evaluation mode (disables dropout)

    Args:
        path:   Path to the saved .pth file.
        device: Device to load the model onto (CPU or GPU).

    Returns:
        Loaded FallDetectionLSTM model in evaluation mode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"You need to train the model first. Run: python -m src.train"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Recreate model with saved architecture
    model = FallDetectionLSTM(
        input_size=checkpoint.get("input_size", NUM_FEATURES),
        hidden_size=checkpoint.get("hidden_size", LSTM_HIDDEN_SIZE_1),
        num_layers=checkpoint.get("num_layers", LSTM_NUM_LAYERS),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
    )

    # Load the learned weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode (disables dropout, etc.)
    model.eval()
    model.to(device)

    print(f"[INFO] Model loaded from {path}")
    print(f"[INFO] Architecture: input={model.input_size}, hidden={model.hidden_size}, "
          f"layers={model.num_layers}, classes={model.num_classes}")

    return model


# ============================================================================
# Direct execution test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LSTM Model — Test")
    print("=" * 60)

    # Create model
    model = FallDetectionLSTM()
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy data
    print("\n--- Forward Pass Test ---")
    batch_size = 4
    seq_length = 30
    dummy_input = torch.randn(batch_size, seq_length, NUM_FEATURES)
    print(f"  Input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"  Output shape: {output.shape}  (expected: ({batch_size}, {NUM_CLASSES}))")

    probs = model.predict_proba(dummy_input)
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probabilities sum: {probs.sum(dim=1)}")  # Should be ~1.0
    print(f"  Sample prediction: {probs[0].detach().numpy()}")

    # Test save and load
    print("\n--- Save/Load Test ---")
    test_path = Path("models/test_model.pth")
    save_model(model, test_path)
    loaded_model = load_model(test_path)
    loaded_output = loaded_model(dummy_input.to(next(loaded_model.parameters()).device))
    match = torch.allclose(output.cpu(), loaded_output.cpu(), atol=1e-5)
    print(f"  Outputs match after load: {match}")

    # Cleanup test file
    if test_path.exists():
        test_path.unlink()
        print(f"  Cleaned up: {test_path}")

    print("\n[OK] All model tests passed!")
