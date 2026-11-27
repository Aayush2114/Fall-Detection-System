# Dataset Directory

## Overview

This directory stores the datasets used by the fall detection system.

## Directory Structure

```
data/
├── raw/        ← Place downloaded dataset files here
├── processed/  ← Automatically generated processed data
└── README.md   ← This file
```

## Recommended Dataset: UP-Fall Detection (3D Skeletons)

### About the Dataset

The **UP-Fall Detection Dataset** was created by researchers at Universidad Panamericana, Mexico. It contains data from 17 subjects performing 11 activities (5 types of falls + 6 daily activities), with 3 trials each.

The **3D Skeletons variant** provides pre-extracted skeleton data using MediaPipe BlazePose, making it directly compatible with our pipeline.

**Citation:**
> Martínez-Villaseñor, L., Ponce, H., Brieva, J., Moya-Albor, E., Núñez-Martínez, J., & Peñafort-Asturiano, C. (2019). UP-Fall Detection Dataset: A Multimodal Approach. Sensors, 19(9), 1988.

### Download Instructions

#### Option A: 3D Skeleton Data (Recommended — Small Download)

1. Visit: https://zenodo.org/records/12781431
2. Download the skeleton data files
3. Extract into `data/raw/`
4. Organize files by subject and activity

#### Option B: Full UP-Fall Dataset (850+ GB — Advanced)

1. Visit: https://sites.google.com/up.edu.mx/har-up/
2. Request access (academic use)
3. Download video files
4. Use our `pose_extraction.py` to extract landmarks

### Expected File Structure After Download

```
data/raw/
├── Subject1/
│   ├── Activity1/
│   │   ├── Trial1.csv
│   │   ├── Trial2.csv
│   │   └── Trial3.csv
│   ├── Activity2/
│   │   └── ...
│   └── Activity11/
├── Subject2/
│   └── ...
└── Subject17/
```

Each CSV should contain frame-by-frame landmark data with at least 132 columns (33 landmarks × 4 values per landmark).

### Activity Labels

| Activity ID | Type | Description |
|-------------|------|-------------|
| 1 | Fall | Falling forward using hands |
| 2 | Fall | Falling forward using knees |
| 3 | Fall | Falling backwards |
| 4 | Fall | Falling sideward |
| 5 | Fall | Falling sitting in chair |
| 6 | Normal | Walking |
| 7 | Normal | Standing |
| 8 | Normal | Sitting |
| 9 | Normal | Picking up objects |
| 10 | Normal | Jumping |
| 11 | Normal | Lying down |

## Synthetic Data (For Pipeline Testing)

If you don't have the real dataset, the system can generate synthetic data for testing purposes:

```bash
python -m src.train --synthetic
```

⚠️ **WARNING:** Synthetic data results do NOT represent real-world model performance. They only verify that the code pipeline works correctly.

## Licensing

- **UP-Fall Dataset:** Available for academic and research use. Please cite the original paper.
- **3D Skeletons variant:** CC BY 4.0 license (free with attribution).

## Data Leakage Prevention

Our data splitting strategy splits by **subject ID**, not randomly:
- Subjects 1–11: Training
- Subjects 12–14: Validation
- Subjects 15–17: Testing

This ensures the model is evaluated on completely unseen individuals.
