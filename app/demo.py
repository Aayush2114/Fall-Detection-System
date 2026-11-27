"""
Fall Detection Demo Application
=================================

A simple command-line interface for demonstrating the fall detection system.

This script provides a user-friendly menu to:
1. Run inference on a video file
2. Run inference on webcam (if available)
3. Train the model (using synthetic or real data)
4. Evaluate the model
5. View system information

This is the main entry point for end-users and demonstrations.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  🏥 AI-Based Human Fall Detection System")
    print("  Using Computer Vision and Deep Learning")
    print("=" * 60)
    print("  MediaPipe Pose + LSTM Classifier")
    print("  Educational Research Prototype")
    print("-" * 60)


def print_menu():
    """Print the main menu."""
    print("\n📋 Main Menu:")
    print("  [1] Run fall detection on a video file")
    print("  [2] Run fall detection on webcam (live)")
    print("  [3] Train model (synthetic data)")
    print("  [4] Train model (real dataset)")
    print("  [5] Evaluate trained model")
    print("  [6] System information")
    print("  [0] Exit")
    print()


def run_video_inference():
    """Run fall detection on a user-specified video."""
    from src.inference import run_inference_on_video
    from src.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH

    video_path = input("Enter video file path: ").strip().strip('"')

    if not Path(video_path).exists():
        print(f"[ERROR] File not found: {video_path}")
        return

    if not MODEL_SAVE_PATH.exists():
        print("[ERROR] No trained model found. Please train the model first (option 3 or 4).")
        return

    if not SCALER_SAVE_PATH.exists():
        print("[ERROR] No scaler found. Please train the model first.")
        return

    print("\n[INFO] Starting inference...")
    try:
        stats = run_inference_on_video(
            video_path=video_path,
            show_display=True,
        )
        print("\n✅ Inference complete!")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")


def run_webcam_inference():
    """Run fall detection on live webcam feed."""
    from src.inference import run_inference_on_webcam
    from src.config import MODEL_SAVE_PATH, SCALER_SAVE_PATH

    if not MODEL_SAVE_PATH.exists():
        print("[ERROR] No trained model found. Please train first (option 3 or 4).")
        return

    if not SCALER_SAVE_PATH.exists():
        print("[ERROR] No scaler found. Please train first.")
        return

    print("\n[INFO] Starting webcam inference. Press 'q' to stop.")
    try:
        run_inference_on_webcam()
    except RuntimeError as e:
        print(f"\n[ERROR] Webcam error: {e}")
    except Exception as e:
        print(f"\n[ERROR] {e}")


def train_synthetic():
    """Train model using synthetic data."""
    from src.train import train_model

    print("\n⚠️  Using SYNTHETIC data for pipeline testing.")
    print("   Results will NOT represent real-world performance.\n")

    confirm = input("Continue? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        return

    try:
        history = train_model(use_synthetic=True, num_epochs=30)
        print("\n✅ Training complete! Model saved.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")


def train_real():
    """Train model using real dataset."""
    from src.train import train_model
    from src.config import RAW_DATA_DIR

    csv_files = list(RAW_DATA_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"\n[ERROR] No data found in {RAW_DATA_DIR}")
        print("Please download the UP-Fall skeleton dataset first.")
        print("See data/README.md for instructions.")
        return

    print(f"\n[INFO] Found {len(csv_files)} data files.")
    confirm = input("Start training? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        return

    try:
        history = train_model(use_synthetic=False)
        print("\n✅ Training complete! Model saved.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")


def run_evaluation():
    """Evaluate the trained model."""
    from src.evaluate import evaluate_model
    from src.config import MODEL_SAVE_PATH

    if not MODEL_SAVE_PATH.exists():
        print("[ERROR] No trained model found. Please train first.")
        return

    try:
        # Check if real data exists; if not, evaluate on synthetic
        from src.config import RAW_DATA_DIR
        csv_files = list(RAW_DATA_DIR.rglob("*.csv"))
        use_synthetic = len(csv_files) == 0

        if use_synthetic:
            print("[INFO] No real dataset found. Evaluating with synthetic data.")
            print("⚠️  Synthetic results do NOT represent real performance.\n")

        metrics = evaluate_model(use_synthetic=use_synthetic)
        print("\n✅ Evaluation complete! Check outputs/ for plots.")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")


def show_system_info():
    """Display system and configuration information."""
    import torch
    from src.config import (
        NUM_FEATURES, SEQUENCE_LENGTH, NUM_CLASSES,
        LSTM_HIDDEN_SIZE_1, LSTM_NUM_LAYERS,
        MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    )

    print("\n📊 System Information:")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {'Available (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available (using CPU)'}")

    print(f"\n🏗️  Model Configuration:")
    print(f"  Input features:  {NUM_FEATURES} (33 landmarks × 4 values)")
    print(f"  Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"  LSTM hidden:     {LSTM_HIDDEN_SIZE_1}")
    print(f"  LSTM layers:     {LSTM_NUM_LAYERS}")
    print(f"  Output classes:  {NUM_CLASSES}")

    print(f"\n📁 Saved Files:")
    print(f"  Model: {MODEL_SAVE_PATH} {'✅' if MODEL_SAVE_PATH.exists() else '❌ Not trained'}")
    print(f"  Scaler: {SCALER_SAVE_PATH} {'✅' if SCALER_SAVE_PATH.exists() else '❌ Not trained'}")


def main():
    """Main application loop."""
    print_banner()

    while True:
        print_menu()
        choice = input("Select option: ").strip()

        if choice == "1":
            run_video_inference()
        elif choice == "2":
            run_webcam_inference()
        elif choice == "3":
            train_synthetic()
        elif choice == "4":
            train_real()
        elif choice == "5":
            run_evaluation()
        elif choice == "6":
            show_system_info()
        elif choice == "0":
            print("\nGoodbye! 👋")
            break
        else:
            print("[ERROR] Invalid option. Please choose 0-6.")


if __name__ == "__main__":
    main()
