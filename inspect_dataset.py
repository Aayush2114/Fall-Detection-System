
import os
import glob
import pandas as pd

RAW_DIR = "data/raw"

total_files = 0
total_frames = 0

print("=" * 65)
print("       UP-FALL REAL DATASET INSPECTION")
print("=" * 65)

for subject in sorted(glob.glob(os.path.join(RAW_DIR, "SUBJECT*"))):

    files = glob.glob(os.path.join(subject, "*.csv"))

    print(f"\n{os.path.basename(subject)}")
    print("-" * 45)

    subject_frames = 0
    subject_labels = {0: 0, 1: 0}

    for file in sorted(files):
        try:
            df = pd.read_csv(file)

            label_col = next(
                (col for col in ["LABEL", "LLABEL"]
                 if col in df.columns),
                None
            )

            coordinate_cols = [
                col for col in df.columns
                if col.startswith("Joint")
                and col.endswith(("_X", "_Y", "_Z"))
            ]

            if label_col is None:
                print(f"[WARNING] {os.path.basename(file)}: No label column")
                continue

            if len(coordinate_cols) != 99:
                print(
                    f"[WARNING] {os.path.basename(file)}: "
                    f"{len(coordinate_cols)} coordinate columns"
                )
                continue

            counts = df[label_col].value_counts().to_dict()

            subject_frames += len(df)

            for label in [0, 1]:
                subject_labels[label] += int(counts.get(label, 0))

            total_files += 1
            total_frames += len(df)

            print(
                f"{os.path.basename(file)} | "
                f"Frames: {len(df)} | "
                f"Labels: {counts}"
            )

        except Exception as e:
            print(f"[ERROR] {os.path.basename(file)}: {e}")

    print(f"\nSubject files: {len(files)}")
    print(f"Valid frames: {subject_frames}")
    print(f"Label totals: {subject_labels}")

print("\n" + "=" * 65)
print("OVERALL SUMMARY")
print("=" * 65)
print("Valid CSV files:", total_files)
print("Total frames:", total_frames)
