"""
Inference Module — Run Fall Detection on New Videos
====================================================

PURPOSE:
    Use the trained LSTM model to analyze new video files or webcam feeds
    and predict whether the person is performing normal activity or falling.

HOW INFERENCE WORKS:
    1. Load the trained model and feature scaler
    2. Open the video
    3. For each frame:
       a. Extract pose landmarks with MediaPipe
       b. Collect landmarks into a sliding window buffer
       c. When buffer has 30 frames, normalize and predict
       d. Display prediction and confidence on the frame
    4. Save annotated output video

DIFFERENCE FROM TRAINING:
    - model.eval() mode — dropout is disabled
    - torch.no_grad() — no gradient computation needed
    - We process frames one at a time (streaming/real-time style)
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
from typing import Optional
from collections import deque

from src.config import (
    NUM_FEATURES, SEQUENCE_LENGTH, NUM_CLASSES,
    CLASS_LABELS, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    NORMAL_COLOR, FALL_COLOR, FONT_SCALE, FONT_THICKNESS,
    OUTPUTS_DIR,
)
from src.model import load_model
from src.preprocessing import load_scaler
from src.utils import get_device

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def run_inference_on_video(
    video_path: str,
    model_path: str = str(MODEL_SAVE_PATH),
    scaler_path: str = str(SCALER_SAVE_PATH),
    output_path: Optional[str] = None,
    show_display: bool = True,
    confidence_threshold: float = 0.6,
) -> dict:
    """
    Run fall detection inference on a video file.

    This function:
    1. Opens the video and loads the trained model
    2. Processes each frame with MediaPipe Pose
    3. Accumulates frames in a sliding window
    4. When enough frames are collected, makes a prediction
    5. Overlays the prediction on the video
    6. Optionally saves the annotated output

    Args:
        video_path:            Path to input video file.
        model_path:            Path to trained model .pth file.
        scaler_path:           Path to fitted scaler .pkl file.
        output_path:           Where to save output video (None = auto-name).
        show_display:          Show live video window.
        confidence_threshold:  Minimum confidence to trigger fall alert.

    Returns:
        Dictionary with inference statistics.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ---- Load Model and Scaler ----
    device = get_device()
    model = load_model(Path(model_path), device)
    scaler = load_scaler(Path(scaler_path))

    # ---- Open Video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input: {video_path.name} ({width}×{height} @ {fps:.0f} FPS)")

    # ---- Setup Output Video ----
    if output_path is None:
        output_path = str(OUTPUTS_DIR / f"output_{video_path.stem}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ---- Sliding Window Buffer ----
    # We collect landmarks frame by frame into this buffer.
    # When it has SEQUENCE_LENGTH frames, we make a prediction.
    landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # ---- Statistics ----
    stats = {
        "total_frames": 0,
        "frames_with_pose": 0,
        "predictions_made": 0,
        "fall_detections": 0,
        "normal_detections": 0,
    }

    # Current prediction state
    current_prediction = "Waiting for data..."
    current_confidence = 0.0
    current_color = (255, 255, 255)  # White

    # ---- Main Processing Loop ----
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            stats["total_frames"] += 1

            # --- Extract Pose ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # --- Draw Skeleton ---
            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Extract landmark vector
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmark_buffer.append(landmarks)
                stats["frames_with_pose"] += 1

            # --- Make Prediction when buffer is full ---
            if len(landmark_buffer) == SEQUENCE_LENGTH:
                # Prepare input
                sequence = np.array(list(landmark_buffer), dtype=np.float32)
                # Normalize using the same scaler as training
                sequence_norm = scaler.transform(sequence)
                # Convert to tensor: (1, seq_len, features)
                input_tensor = torch.tensor(
                    sequence_norm, dtype=torch.float32
                ).unsqueeze(0).to(device)

                # Predict
                with torch.no_grad():
                    probs = model.predict_proba(input_tensor)
                    pred_class = torch.argmax(probs, dim=1).item()
                    pred_confidence = probs[0, pred_class].item()

                current_prediction = CLASS_LABELS[pred_class]
                current_confidence = pred_confidence
                current_color = FALL_COLOR if pred_class == 1 else NORMAL_COLOR

                stats["predictions_made"] += 1
                if pred_class == 1:
                    stats["fall_detections"] += 1
                else:
                    stats["normal_detections"] += 1

            # --- Draw Prediction Overlay ---
            # Background box for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 100), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Prediction text
            cv2.putText(
                frame, f"Status: {current_prediction}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, current_color, FONT_THICKNESS,
            )
            cv2.putText(
                frame, f"Confidence: {current_confidence:.1%}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1,
            )

            # Fall alert banner
            if current_prediction == "Potential Fall" and current_confidence >= confidence_threshold:
                cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 200), -1)
                cv2.putText(
                    frame, "⚠ FALL DETECTED — ALERT",
                    (width // 2 - 180, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                )

            # Write output frame
            writer.write(frame)

            # Show live display
            if show_display:
                cv2.imshow("Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Display stopped by user (q).")
                    break

            # Progress
            if stats["total_frames"] % 200 == 0:
                pct = stats["total_frames"] / total_frames * 100
                print(f"  Progress: {stats['total_frames']}/{total_frames} ({pct:.0f}%)")

    # Cleanup
    cap.release()
    writer.release()
    if show_display:
        cv2.destroyAllWindows()

    print(f"\n[INFO] Inference complete!")
    print(f"  Output saved: {output_path}")
    print(f"  Frames processed: {stats['total_frames']}")
    print(f"  Frames with pose: {stats['frames_with_pose']}")
    print(f"  Predictions made: {stats['predictions_made']}")
    print(f"  Fall detections: {stats['fall_detections']}")
    print(f"  Normal detections: {stats['normal_detections']}")

    return stats


def run_inference_on_webcam(
    model_path: str = str(MODEL_SAVE_PATH),
    scaler_path: str = str(SCALER_SAVE_PATH),
    confidence_threshold: float = 0.6,
) -> None:
    """
    Run fall detection on live webcam feed.

    Press 'q' to stop, 's' to save a screenshot.

    Args:
        model_path:            Path to trained model.
        scaler_path:           Path to fitted scaler.
        confidence_threshold:  Minimum confidence for fall alert.
    """
    device = get_device()
    model = load_model(Path(model_path), device)
    scaler = load_scaler(Path(scaler_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = "Waiting..."
    current_confidence = 0.0
    current_color = (255, 255, 255)

    print("[INFO] Webcam inference started. Press 'q' to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                landmark_buffer.append(landmarks)

            if len(landmark_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(list(landmark_buffer), dtype=np.float32)
                sequence_norm = scaler.transform(sequence)
                input_tensor = torch.tensor(
                    sequence_norm, dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = model.predict_proba(input_tensor)
                    pred_class = torch.argmax(probs, dim=1).item()
                    pred_confidence = probs[0, pred_class].item()

                current_prediction = CLASS_LABELS[pred_class]
                current_confidence = pred_confidence
                current_color = FALL_COLOR if pred_class == 1 else NORMAL_COLOR

            # Overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 90), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            cv2.putText(frame, f"{current_prediction}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)
            cv2.putText(frame, f"Confidence: {current_confidence:.1%}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if current_prediction == "Potential Fall" and current_confidence >= confidence_threshold:
                cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
                cv2.putText(frame, "FALL DETECTED", (w // 2 - 100, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Fall Detection - Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ss_path = str(OUTPUTS_DIR / "webcam_screenshot.png")
                cv2.imwrite(ss_path, frame)
                print(f"[INFO] Screenshot saved: {ss_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam inference stopped.")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fall Detection Inference")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video (omit for webcam)"
    )
    parser.add_argument(
        "--model", type=str, default=str(MODEL_SAVE_PATH),
        help="Path to trained model"
    )
    parser.add_argument(
        "--scaler", type=str, default=str(SCALER_SAVE_PATH),
        help="Path to feature scaler"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output video path"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't show live display window"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Confidence threshold for fall alert (default: 0.6)"
    )
    args = parser.parse_args()

    if args.video:
        run_inference_on_video(
            video_path=args.video,
            model_path=args.model,
            scaler_path=args.scaler,
            output_path=args.output,
            show_display=not args.no_display,
            confidence_threshold=args.threshold,
        )
    else:
        run_inference_on_webcam(
            model_path=args.model,
            scaler_path=args.scaler,
            confidence_threshold=args.threshold,
        )
