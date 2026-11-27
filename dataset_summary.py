
import os
import glob
import pandas as pd

for subject in sorted(glob.glob("data/raw/SUBJECT*")):
    files = glob.glob(os.path.join(subject, "*.csv"))

    total_frames = 0
    labels = {0: 0, 1: 0}
    valid_files = 0

    for file in files:
        df = pd.read_csv(file)

        label_col = next(
            (c for c in ["LABEL", "LLABEL"] if c in df.columns),
            None
        )

        if label_col is None:
            continue

        features = [
            c for c in df.columns
            if c.startswith("Joint")
            and c.endswith(("_X", "_Y", "_Z"))
        ]

        if len(features) != 99:
            continue

        valid_files += 1
        total_frames += len(df)

        counts = df[label_col].value_counts()
        labels[0] += int(counts.get(0, 0))
        labels[1] += int(counts.get(1, 0))

    print(f"\n{os.path.basename(subject)}")
    print("CSV files:", valid_files)
    print("Frames:", total_frames)
    print("Label 0:", labels[0])
    print("Label 1:", labels[1])
