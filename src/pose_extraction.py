"""
Human Pose Extraction Module
=============================

PURPOSE:
    Extract human body landmark positions from video using MediaPipe Pose.

HOW IT WORKS:
    1. Open a video file with OpenCV
    2. Read each frame (image) one by one
    3. Run MediaPipe Pose on each frame to find the person
    4. MediaPipe returns 33 body landmarks (joints like nose, shoulders, hips, etc.)
    5. Each landmark has 4 values: x, y, z position + visibility confidence
    6. We flatten these into a vector of 33 × 4 = 132 numbers per frame
    7. Collect all frame vectors into a 2D array

WHAT ARE LANDMARKS?
    MediaPipe Pose identifies 33 key points on the human body:
    - 0: Nose
    - 11, 12: Left/Right Shoulder
    - 13, 14: Left/Right Elbow
    - 15, 16: Left/Right Wrist
    - 23, 24: Left/Right Hip
    - 25, 26: Left/Right Knee
    - 27, 28: Left/Right Ankle
    ... and more (see MediaPipe documentation)

    Each landmark is normalized to [0, 1] relative to the frame dimensions.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from src.config import (
    NUM_LANDMARKS, LANDMARK_FEATURES, NUM_FEATURES,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)

# ============================================================================
# Initialize MediaPipe Pose components
# ============================================================================
# mp.solutions.pose      — The pose estimation model
# mp.solutions.drawing_utils — Helper to draw landmarks on images
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_landmarks_from_frame(
    frame: np.ndarray,
    pose_model,
) -> Tuple[Optional[np.ndarray], object]:
    """
    Extract pose landmarks from a single video frame.

    Steps:
        1. Convert BGR → RGB (MediaPipe expects RGB)
        2. Run pose detection
        3. If person found → return 132-dim feature vector
        4. If no person → return None

    Args:
        frame:      Single video frame, shape (H, W, 3), BGR color.
        pose_model: Initialized MediaPipe Pose instance.

    Returns:
        (landmarks_vector, results)
        landmarks_vector: np.ndarray of shape (132,) or None.
        results:          Raw MediaPipe results (for drawing).
    """
    # MediaPipe needs RGB; OpenCV reads BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Setting writeable=False tells MediaPipe it can skip copying the array
    # (a small performance optimization)
    rgb_frame.flags.writeable = False
    results = pose_model.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # No person detected in this frame
    if results.pose_landmarks is None:
        return None, results

    # Extract all 33 landmarks into a flat list
    # Each landmark → [x, y, z, visibility]
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    # Convert to numpy array: shape (132,)
    return np.array(landmarks, dtype=np.float32), results


def draw_landmarks_on_frame(
    frame: np.ndarray,
    results,
) -> np.ndarray:
    """
    Draw detected pose skeleton on the frame.

    Draws:
    - Colored dots at each detected joint
    - Lines connecting related joints (the skeleton)

    Args:
        frame:   Original video frame (BGR).
        results: MediaPipe Pose detection results.

    Returns:
        Copy of frame with landmarks drawn on it.
    """
    annotated = frame.copy()
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated


def extract_landmarks_from_video(
    video_path: str,
    save_annotated: bool = False,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[int], dict]:
    """
    Process an entire video and extract pose landmarks from every frame.

    This is the MAIN extraction function. It:
    1. Opens the video file
    2. Reads each frame sequentially
    3. Runs pose detection on each frame
    4. Collects valid landmark vectors
    5. Optionally saves an annotated video with skeleton overlay

    Args:
        video_path:     Path to input video file.
        save_annotated: If True, save a video with skeleton drawn on it.
        output_path:    Where to save the annotated video.
        max_frames:     Process at most this many frames (None = all).

    Returns:
        Tuple of:
        - landmarks_array: np.ndarray, shape (N, 132), where N = frames with detection
        - valid_frame_indices: list of frame numbers that had a detected person
        - video_info: dict with metadata (fps, resolution, etc.)

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        ValueError: If the video file can't be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            f"Please check the file path and try again."
        )

    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}\n"
            f"The file may be corrupted or in an unsupported format."
        )

    # Read video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_info = {
        "path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": round(total_frames / fps, 2) if fps > 0 else 0,
    }

    print(f"[INFO] Processing: {video_path.name}")
    print(
        f"[INFO] {width}×{height} @ {fps:.0f} FPS, "
        f"{total_frames} frames, {video_info['duration_seconds']}s"
    )

    # Optional: set up video writer for annotated output
    writer = None
    if save_annotated and output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ---- Main extraction loop ----
    all_landmarks = []        # List of 132-dim vectors
    valid_frame_indices = []  # Which frame numbers had a person
    frames_processed = 0

    with mp_pose.Pose(
        static_image_mode=False,          # Video mode: uses tracking
        model_complexity=1,               # 0=lite, 1=full, 2=heavy
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Extract landmarks from this frame
            lm_vector, results = extract_landmarks_from_frame(frame, pose)

            if lm_vector is not None:
                all_landmarks.append(lm_vector)
                valid_frame_indices.append(frames_processed)

            # Save annotated frame if requested
            if writer is not None:
                annotated = draw_landmarks_on_frame(frame, results)
                writer.write(annotated)

            frames_processed += 1

            # Progress update every 200 frames
            if frames_processed % 200 == 0:
                pct = frames_processed / total_frames * 100
                print(f"  Progress: {frames_processed}/{total_frames} ({pct:.0f}%)")

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Annotated video saved: {output_path}")

    # Convert collected landmarks to a numpy array
    if len(all_landmarks) == 0:
        print("[WARNING] No person detected in any frame!")
        landmarks_array = np.empty((0, NUM_FEATURES), dtype=np.float32)
    else:
        landmarks_array = np.array(all_landmarks, dtype=np.float32)

    detection_rate = (
        len(valid_frame_indices) / frames_processed * 100
        if frames_processed > 0
        else 0
    )
    print(
        f"[INFO] Done: {len(valid_frame_indices)}/{frames_processed} frames "
        f"with pose ({detection_rate:.1f}%)"
    )
    print(f"[INFO] Output array shape: {landmarks_array.shape}")

    return landmarks_array, valid_frame_indices, video_info


def extract_landmarks_from_webcam(
    duration_seconds: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """
    Capture pose landmarks from a live webcam feed.

    Opens the default camera, shows the skeleton overlay in real-time,
    and collects landmark data for the specified duration.
    Press 'q' to stop early.

    Args:
        duration_seconds: How many seconds to capture.

    Returns:
        (landmarks_array, video_info)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam.\n"
            "Make sure a camera is connected and not used by another app."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(duration_seconds * fps)
    all_landmarks = []
    frames_processed = 0

    print(f"[INFO] Webcam capture: {duration_seconds}s. Press 'q' to stop.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            lm_vector, results = extract_landmarks_from_frame(frame, pose)
            if lm_vector is not None:
                all_landmarks.append(lm_vector)

            # Show live feed with skeleton
            annotated = draw_landmarks_on_frame(frame, results)
            cv2.imshow("Fall Detection - Webcam (press 'q' to stop)", annotated)

            frames_processed += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

    landmarks_array = (
        np.array(all_landmarks, dtype=np.float32)
        if all_landmarks
        else np.empty((0, NUM_FEATURES), dtype=np.float32)
    )

    video_info = {
        "source": "webcam",
        "frames_processed": frames_processed,
        "frames_with_pose": len(all_landmarks),
        "fps": fps,
    }
    print(f"[INFO] Webcam done: {len(all_landmarks)} frames with pose.")
    return landmarks_array, video_info


# ============================================================================
# Direct execution test
# ============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Pose Extraction Module — Test")
    print("=" * 60)

    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        landmarks, indices, info = extract_landmarks_from_video(video_file)
        print(f"\nResults:")
        print(f"  Shape: {landmarks.shape}")
        print(f"  Frames with pose: {len(indices)}")
        if landmarks.size > 0:
            print(f"  Value range: [{landmarks.min():.4f}, {landmarks.max():.4f}]")
    else:
        print("\nUsage:")
        print("  python -m src.pose_extraction <video_path>")
        print("\nNo video provided. Testing webcam for 5 seconds...")
        try:
            landmarks, info = extract_landmarks_from_webcam(duration_seconds=5)
            print(f"  Webcam shape: {landmarks.shape}")
        except RuntimeError as e:
            print(f"  Webcam unavailable: {e}")
