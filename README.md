# AI-Based Human Fall Detection System Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

An educational project exploring human fall-event detection from
temporal 3D skeleton data using a Long Short-Term Memory (LSTM) neural
network.

> **Current status:** The development environment, LSTM model, and
> synthetic-data training/evaluation pipeline have been tested. The real
> dataset has been downloaded and inspected, but real-data
> preprocessing, training, and evaluation are still in progress. No
> real-dataset performance is claimed.

> **Disclaimer:** This is an academic prototype, not a medically
> validated fall-detection or emergency-alert system.

------------------------------------------------------------------------

## Contents

-   [Project Overview](#project-overview)
-   [Current Development Progress](#current-development-progress)
-   [Dataset](#dataset)
-   [Dataset Format](#dataset-format)
-   [Planned Methodology](#planned-methodology)
-   [Model](#model)
-   [Environment and Technologies](#environment-and-technologies)
-   [Repository Structure](#repository-structure)
-   [Setup](#setup)
-   [Synthetic Pipeline Test](#synthetic-pipeline-test)
-   [Real Dataset Pipeline](#real-dataset-pipeline)
-   [Evaluation Plan](#evaluation-plan)
-   [Limitations](#limitations)
-   [Roadmap](#roadmap)
-   [Dataset Attribution](#dataset-attribution)

------------------------------------------------------------------------

## Project Overview

Falls can cause serious injuries, particularly among older adults. This
project explores whether temporal patterns in 3D human joint coordinates
can help distinguish impact events from non-impact movement.

The downloaded dataset contains 3D skeleton coordinates extracted from
pose-estimation output. Each frame contains 33 joints, with three
coordinate values (`x`, `y`, `z`) per joint. The intended real-data
model input therefore has **99 features per frame**.

The project uses an LSTM because movement is temporal: a sequence of
body poses can contain information that is not available from a single
frame.

The current codebase also contains a separate MediaPipe-oriented
representation with 33 landmarks and four values (`x`, `y`, `z`,
`visibility`), or **132 features per frame**. This is a different input
representation. The existing 132-feature synthetic test pipeline must
not be confused with the 99-feature downloaded dataset pipeline.

## Current Development Progress

This table reflects the work completed so far.

  -----------------------------------------------------------------------
  Component               Status                  Notes
  ----------------------- ----------------------- -----------------------
  Python environment and  Complete                Python 3.13.7; core
  dependencies                                    library imports
                                                  verified

  GPU environment         Verified                NVIDIA GTX 1650 4 GB;
                                                  CUDA available in the
                                                  tested environment

  LSTM model              Implemented and tested  Model accepts
  implementation                                  configurable input
                                                  size; default is 132

  Synthetic data          Tested                  400 sequences, each
  generation                                      with 30 frames and 132
                                                  features

  Synthetic training      Tested                  30-epoch run completed;
                                                  synthetic
                                                  train/validation
                                                  accuracy reached 100%

  Synthetic evaluation    Tested                  60-sequence synthetic
                                                  test split; 100%
                                                  accuracy in that
                                                  synthetic test

  Real dataset download   Complete                Five subject ZIP
                                                  archives downloaded and
                                                  extracted

  Real dataset inspection Complete                81 CSV files and 8,100
                                                  frames inspected

  Real-data loader        In progress             Existing loader
                                                  assumptions do not
                                                  match downloaded
                                                  folder/CSV format

  Frame-level sequence    In progress             Sequence-label strategy
  labels                                          still needs
                                                  implementation and
                                                  validation

  Subject-wise real-data  Pending                 Must be designed around
  split                                           five available subjects

  Real-data model         Not started             No real-data model
  training                                        performance is
                                                  available

  Real-data evaluation    Not started             No real-data accuracy,
                                                  precision, recall or
                                                  F1-score is claimed

  Video/webcam            Not verified            End-to-end inference
  demonstration                                   with a
                                                  real-data-trained model
                                                  remains future work

  Final documentation     In progress             Documentation is being
                                                  aligned with the actual
                                                  dataset and
                                                  implementation
  -----------------------------------------------------------------------

### Interpreting the synthetic results

The 100% synthetic accuracy only shows that the model-training and
evaluation code can learn the artificial patterns generated by the
synthetic-data function. It does **not** demonstrate real-world
fall-detection accuracy and must not be reported as a real dataset
result.

## Dataset

This project uses the **3D Skeletons UP-Fall Dataset**, hosted on
Zenodo:

**Dataset page:** https://zenodo.org/records/12773013

The dataset description states that it contains 3D skeletal data from
five subjects performing fall scenarios and daily activities. Skeletons
were extracted from image frames using a pose-estimation algorithm.
Recordings with fewer than 100 extracted frames were excluded, which
explains why subjects may have different numbers of CSV files.

### Dataset summary

  --------------------------------------------------------------------------------------------------------
  Attribute                           Description
  ----------------------------------- --------------------------------------------------------------------
  Dataset name                        3D Skeletons UP-Fall Dataset

  Repository                          Zenodo

  DOI                                 [10.5281/zenodo.12773013](https://doi.org/10.5281/zenodo.12773013)

  Subjects                            5

  Downloaded archives                 `SUBJECT1.zip` through `SUBJECT5.zip`

  Data format                         CSV

  Skeleton representation             33 joints × 3 coordinates

  Coordinate features                 99 per frame

  Label column                        `LABEL` (some inspected files use `LLABEL`)

  Label meaning                       `0` = non-impact; `1` = impact

  Licence                             Creative Commons Attribution 4.0 International (CC BY 4.0)
  --------------------------------------------------------------------------------------------------------

### Local dataset inspection

The downloaded files were extracted under:

``` text
data/
└── raw/
    ├── SUBJECT1/
    ├── SUBJECT2/
    ├── SUBJECT3/
    ├── SUBJECT4/
    └── SUBJECT5/
```

Inspection of the local copy produced this summary:

  Subject folder     CSV files      Frames     Label 0     Label 1
  ---------------- ----------- ----------- ----------- -----------
  SUBJECT1                  20       2,000         540       1,460
  SUBJECT2                  21       2,100         689       1,411
  SUBJECT3                  19       1,900         670       1,230
  SUBJECT4                   5         500         180         320
  SUBJECT5                  16       1,600         429       1,171
  **Total**             **81**   **8,100**   **2,508**   **5,592**

These counts describe the local dataset inspection at the time of
writing. They are not model results.

### CSV naming convention

The dataset documentation describes filenames using a pattern such as
`C1S1A1T1.csv`:

-   `C`: camera identifier
-   `S`: subject identifier
-   `A`: activity identifier
-   `T`: trial identifier

Some local filenames may contain underscores. A loader should parse
filenames robustly and should not assume nested `Activity` directories.

### CSV columns and labels

The dataset documentation describes three coordinates per joint and a
final impact label. A typical inspected file has 99 coordinate columns
plus one label column, for 100 columns total.

-   `LABEL = 0`: non-impact
-   `LABEL = 1`: impact

Some locally inspected files use `LLABEL` instead of `LABEL`; the data
loader needs to support both names.

**Important:** Impact detection and whole-fall recognition are not
automatically identical tasks. The dataset label identifies impact
versus non-impact. The project must define and validate how frame-level
impact labels are converted into sequence-level targets before training.

### Dataset availability and attribution

The raw dataset is not included in this repository. Download it from the
original Zenodo record and follow the licence and attribution
requirements before redistributing or adapting dataset material.

## Dataset Format

The downloaded real dataset and the MediaPipe representation used in the
synthetic pipeline have different feature dimensions.

  ------------------------------------------------------------------------
  Pipeline              Features per joint              Features per frame
  --------------------- --------------------- ----------------------------
  Downloaded 3D         `x`, `y`, `z`                      33 × 3 = **99**
  skeleton CSVs                               

  MediaPipe Pose        `x`, `y`, `z`,                    33 × 4 = **132**
  representation        `visibility`          
  ------------------------------------------------------------------------

The real-data pipeline must use 99 input features unless a deliberate
feature transformation is added. The default 132-feature configuration
is retained for the existing MediaPipe/synthetic workflow.

## Planned Methodology

The intended real-data workflow is:

1.  **Load CSV recordings:** Read subject folders and CSV files. Accept
    the observed `LABEL` and `LLABEL` column names.
2.  **Validate data:** Check coordinate columns, label values, missing
    values and recording lengths.
3.  **Identify subjects:** Extract subject identifiers from filenames or
    verified metadata.
4.  **Split by subject:** Assign complete subjects to training,
    validation and test partitions before generating windows. This helps
    prevent frames from the same person appearing in multiple
    partitions.
5.  **Normalize features:** Fit the feature scaler on training data
    only, then apply it to validation and test data.
6.  **Create temporal windows:** Convert recordings into fixed-length
    windows (current project setting: 30 frames, stride: 15).
7.  **Assign sequence labels:** Define and document a justified strategy
    for converting frame-level impact labels into sequence-level
    targets. This step is not finalized.
8.  **Train the LSTM:** Train a model configured for 99 input features.
9.  **Evaluate:** Report metrics on held-out subjects only.

The subject split and sequence-label strategy must be finalized before
real-data training. The old 17-subject split and activity-number-based
class mapping do not apply to this five-subject CSV dataset.

## Model

The current model implementation is a stacked LSTM classifier in
PyTorch.

The model class supports a configurable `input_size`:

-   Existing synthetic/MediaPipe pipeline: `input_size=132`
-   Planned real CSV pipeline: `input_size=99`

The current architecture uses a two-layer LSTM with a hidden size of
128, followed by a fully connected classification head. The model
produces two class logits.

The exact real-data training configuration, sequence-label definition
and final checkpoint path will be documented after that pipeline is
implemented and tested.

## Environment and Technologies

Development environment used for the current checks:

  Component           Verified version / detail
  ------------------- --------------------------------
  Python              3.13.7
  PyTorch             2.11.0+cu128
  OpenCV              5.1.0
  MediaPipe           1.2.3
  NumPy               2.5.4
  Pandas              3.1.1
  scikit-learn        1.9.1
  Matplotlib          3.11.2
  tqdm                4.70.1
  GPU                 NVIDIA GeForce GTX 1650, 4 GB
  CUDA availability   True in the tested environment

Main technologies:

-   **Python:** project implementation
-   **PyTorch:** LSTM model and training workflow
-   **Pandas / NumPy:** CSV handling and numerical preprocessing
-   **scikit-learn:** feature scaling and evaluation metrics
-   **OpenCV / MediaPipe:** pose and video tools being explored for a
    separate video-input workflow
-   **Matplotlib:** training and evaluation plots

Versions above describe the development environment used for testing.
For reproducibility, consult `requirements.txt`.

## Repository Structure

The repository is being developed with the following structure. Some
components may still be under development.

``` text
fall-detection-system/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Downloaded subject CSVs (not committed)
│   └── processed/            # Generated preprocessing artifacts
├── notebooks/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── pose_extraction.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── app/
│   └── demo.py
├── models/                   # Model checkpoints (not necessarily committed)
├── outputs/                  # Plots, logs and evaluation reports
└── docs/
    ├── PROJECT_EXPLANATION.md
    ├── TECHNICAL_CONCEPTS.md
    ├── INSTALLATION_GUIDE.md
    └── RESULTS_AND_LIMITATIONS.md
```

## Setup

### Requirements

-   Python (Python 3.13.7 was used for current development checks)
-   Git
-   pip
-   Optional: compatible NVIDIA GPU and CUDA-enabled PyTorch for GPU
    acceleration

### Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fall-detection-system
```

Replace the placeholder with your actual GitHub repository URL.

### Create a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation varies by operating system and CUDA support. If
needed, use the official selector:
https://pytorch.org/get-started/locally/.

### Verify key imports

``` bash
python -c "import torch, cv2, mediapipe, numpy, pandas, sklearn; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Synthetic Pipeline Test

The synthetic mode checks the existing model-training workflow without
the real dataset.

``` bash
python -m src.train --synthetic
python -m src.evaluate --synthetic
```

In the completed synthetic test, the generator created:

-   200 synthetic normal sequences
-   200 synthetic fall-like sequences
-   400 sequences total
-   Sequence shape: `(400, 30, 132)`
-   Train/validation/test split: 280 / 60 / 60

A 30-epoch synthetic training run completed, and the synthetic test
evaluation reported 100% accuracy. These results are only a
software-pipeline check on artificially generated patterns. They are not
evidence of real-data performance, generalization to unseen people, or
reliable fall detection.

## Real Dataset Pipeline

The real CSV dataset has been downloaded and inspected, but the
real-data loader and training path still need adaptation.

Before real-data training, the project needs to:

-   Load the five `SUBJECT` folders and parse filename metadata.
-   Select the 99 coordinate columns, not the 132-feature MediaPipe
    representation.
-   Accept both `LABEL` and `LLABEL` where present.
-   Preserve frame-level impact labels.
-   Establish a subject-wise train/validation/test split using the five
    available subjects.
-   Finalize and document the sequence-level label rule.
-   Fit preprocessing only on training subjects.
-   Instantiate the LSTM with 99 input features.
-   Save the real-data checkpoint and scaler separately from
    synthetic/MediaPipe artifacts.

**Do not treat `python -m src.train` as a verified real-data training
command yet.** The real-data path is still being adapted and must be
tested after these changes.

## Evaluation Plan

Once real-data training is implemented, evaluation should use subjects
excluded from training and validation.

Planned metrics:

-   **Accuracy:** overall fraction of correct predictions
-   **Precision:** fraction of predicted impact events that are actual
    impact events
-   **Recall:** fraction of actual impact events detected
-   **F1-score:** harmonic mean of precision and recall
-   **Confusion matrix:** counts of true positives, true negatives,
    false positives and false negatives

Because the dataset has only five subjects and recording counts vary by
subject, results should be interpreted cautiously. Subject-wise
evaluation is important to reduce data leakage, and the small
test-subject count limits generalization claims.

No real-data metric is reported here because real-data training and
evaluation have not yet been completed.

## Limitations

1.  **Small number of subjects:** The downloaded dataset contains five
    subjects. This limits diversity and the strength of generalization
    claims.
2.  **Impact labels:** The provided labels indicate impact versus
    non-impact. They should not be described as fully validated
    fall/no-fall labels without a clearly defined task and validation.
3.  **Sequence-label design:** Frame-level impact labels need a
    carefully justified conversion to sequence labels. Different
    window-label rules can produce different learning targets.
4.  **Controlled data:** The dataset contains staged scenarios and may
    not represent falls in real homes, hospitals or care facilities.
5.  **Camera and coordinate sensitivity:** 3D joint coordinates can vary
    with camera placement, pose-estimation quality and recording
    conditions.
6.  **Synthetic data:** Synthetic results are not representative of real
    performance.
7.  **Not medically validated:** The system is an educational prototype
    and is not suitable for clinical or emergency use.

## Roadmap

-   [x] Set up Python environment and install core dependencies
-   [x] Verify PyTorch and CUDA availability
-   [x] Implement and smoke-test the LSTM model
-   [x] Run synthetic training and evaluation
-   [x] Download the 3D Skeletons UP-Fall dataset
-   [x] Inspect CSV structure, frame counts and label distribution
-   [ ] Adapt the real-data CSV loader
-   [ ] Finalize frame-level to sequence-level label handling
-   [ ] Define a subject-wise split for five subjects
-   [ ] Implement the 99-feature real-data training path
-   [ ] Train and evaluate on held-out subjects
-   [ ] Add real-data results and plots
-   [ ] Verify video/webcam inference end to end
-   [ ] Complete documentation and demonstration

## Dataset Attribution

This project uses the **3D Skeletons UP-Fall Dataset** published on
Zenodo.

-   Dataset record: https://zenodo.org/records/12773013
-   DOI: https://doi.org/10.5281/zenodo.12773013
-   Creator listed on Zenodo: Tresor Koffi
-   Licence listed on Zenodo: Creative Commons Attribution 4.0
    International (CC BY 4.0)

Consult the original Zenodo record for the dataset description, licence
terms and attribution details. The original dataset files are not
included in this repository.

------------------------------------------------------------------------

**Project status:** In development. This README will be updated as the
real-data pipeline, evaluation and demonstration are completed.
