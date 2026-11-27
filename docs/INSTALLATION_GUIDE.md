# Windows Installation Guide — AI-Based Human Fall Detection System

A step-by-step, test-focused setup guide for **Windows 10 / 11**, written for 2nd-year CS students. Every step includes the exact commands to run and how to verify it worked before moving on.

---

## Prerequisites

Make sure you have (or are ready to install) the following:

- **Python 3.10+** (tested with **Python 3.13**)
- **Git** — for cloning the repository ([git-scm.com/downloads](https://git-scm.com/downloads))
  - Quick check if already installed:
    ```cmd
    git --version
    ```
- **NVIDIA GPU** *(recommended, not required)* — speeds up model training, but all tests, synthetic training, and inference also run on CPU.
  - Check if you have an NVIDIA GPU and working driver:
    ```cmd
    nvidia-smi
    ```
- **VS Code** *(recommended editor)* — ([code.visualstudio.com](https://code.visualstudio.com/))

> **Tip (Command Prompt vs. PowerShell):** The commands below work in **Command Prompt (`cmd`)** and **Windows PowerShell** unless noted otherwise. Where PowerShell differs (such as activating a virtual environment), both options are shown.

---

## Step 1: Install Python

1. Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/) and download the **Windows installer (64-bit)** for **Python 3.10 or newer** (tested with 3.13).
2. Double-click the downloaded `.exe` installer.
3. **CRITICAL:** At the bottom of the first installer screen, **check the box**:
   - ✅ **"Add python.exe to PATH"** (or **"Add Python to PATH"**)
4. Click **Install Now** and finish the setup.
5. Open a **new** Command Prompt or PowerShell window (important: old terminals will not see the updated `PATH`) and verify:

```cmd
python --version
```

Expected output (your patch version may vary):

```text
Python 3.13.7
```

Also verify `pip` is available:

```cmd
pip --version
```

---

## Step 2: Clone or Download the Project

### Option A: Clone with Git (Recommended)

Open Command Prompt or PowerShell in the folder where you keep your projects (for example, `Documents` or `Downloads`), then run:

```cmd
git clone <repository-url>
cd fall-detection-system
```

*(Replace `<repository-url>` with the URL of your Git repository.)*

### Option B: Download ZIP (Without Git)

1. Open the repository page in your browser and click **Code → Download ZIP**.
2. Right-click the downloaded `.zip` file → **Extract All...** and extract it to a folder without spaces if possible (e.g., `C:\Projects\fall-detection-system`).
3. Open Command Prompt or PowerShell and navigate into the extracted folder:

```cmd
cd path\to\fall-detection-system
```

### Verify You Are in the Right Folder

Run `dir` and confirm you see `requirements.txt`, `src`, `app`, and `data`:

```cmd
dir
```

---

## Step 3: Create a Virtual Environment

### What Is a Virtual Environment and Why Use One?

A **virtual environment** (`venv`) is an isolated Python folder just for this project. Without one, `pip install` installs packages globally into your system Python, which can break other coursework or projects that need different library versions (like an older PyTorch or NumPy). With a `venv`, everything stays cleanly contained inside `fall-detection-system\venv\`.

### Create the Virtual Environment

Make sure you are inside the `fall-detection-system` project root folder, then run:

```cmd
python -m venv venv
```

This creates a new `venv\` folder inside your project directory (wait a few seconds for it to finish).

### Activate the Virtual Environment

- **In Command Prompt (`cmd`):**

  ```cmd
  venv\Scripts\activate
  ```

- **In PowerShell:**

  ```powershell
  venv\Scripts\Activate.ps1
  ```

  *(If PowerShell blocks the script with an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once and try `venv\Scripts\Activate.ps1` again — or simply use Command Prompt.)*

### How to Tell It Is Activated

Your terminal prompt will change to show **`(venv)`** at the start of the line:

```text
(venv) C:\Users\YourName\fall-detection-system>
```

You can also confirm Python is pointing inside `venv`:

```cmd
where python
```

The first line printed should end with `fall-detection-system\venv\Scripts\python.exe`.

> **Important:** You must activate `venv` every time you open a new terminal before running project commands.

---

## Step 4: Install PyTorch with CUDA Support

Install PyTorch **before** `requirements.txt` so that `pip` installs the GPU-enabled (CUDA) build instead of falling back to the CPU-only wheel.

1. Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
2. Use the configurator on that page and select:
   - **PyTorch Build:** `Stable`
   - **Your OS:** `Windows`
   - **Package:** `Pip`
   - **Language:** `Python`
   - **Compute Platform:** Choose your **CUDA** version (e.g., `CUDA 12.8`) if you have an NVIDIA GPU, or `CPU` if you do not.
     *(Tip: run `nvidia-smi` in a terminal to see the maximum CUDA version your NVIDIA driver supports in the top-right corner.)*

### Option A: With NVIDIA GPU (CUDA)

Make sure `(venv)` is active, then run the command from PyTorch's site (example for CUDA 12.8):

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Option B: CPU-Only (No NVIDIA GPU)

Make sure `(venv)` is active, then run:

```cmd
pip install torch torchvision torchaudio
```

### Verify PyTorch & CUDA

Run this one-line test:

```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

- Prints `True` → PyTorch installed with working GPU/CUDA support.
- Prints `False` → PyTorch installed in CPU mode (expected if you chose Option B; if you have an NVIDIA GPU and expected `True`, see [Troubleshooting](#troubleshooting)).

Optional detailed check:

```cmd
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Step 5: Install Other Dependencies

With `(venv)` still active, upgrade `pip` first (helps avoid wheel resolution errors on Windows), then install the remaining dependencies from `requirements.txt`:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs OpenCV (`opencv-python`), MediaPipe (`mediapipe`), NumPy, Pandas, Matplotlib, scikit-learn, tqdm, and Jupyter.

### Verify Installations

Check the installed packages list:

```cmd
pip list
```

Confirm key packages (`torch`, `torchvision`, `torchaudio`, `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`) appear without errors.

---

## Step 6: Download the Dataset (Optional)

You can train and evaluate the system on the **UP-Fall Detection (3D Skeletons)** dataset, or skip the download for now and test everything immediately with synthetic data.

### Option A: Download the Real Dataset (UP-Fall 3D Skeletons)

1. Visit the Zenodo record: [https://zenodo.org/records/12781431](https://zenodo.org/records/12781431)
2. Download the 3D skeleton dataset archive.
3. Extract the files into `data\raw\` inside your project folder so the directory layout matches:

```text
data/
└── raw/
    ├── Subject1/
    │   ├── Activity1/
    │   │   ├── Trial1.csv
    │   │   ├── Trial2.csv
    │   │   └── Trial3.csv
    │   └── ...
    ├── Subject2/
    │   └── ...
    └── Subject17/
```

See [data/README.md](../data/README.md) for full dataset details and activity labels.

### Option B: Skip Download & Use `--synthetic` Flag (Fast Testing)

If you want to verify the entire training and evaluation pipeline without downloading any dataset, use the built-in `--synthetic` flag:

```cmd
python -m src.train --synthetic --epochs 5
python -m src.evaluate --synthetic
```

> **Note:** Synthetic data is generated solely to verify that the code pipeline runs end-to-end. Metrics produced with `--synthetic` do **not** reflect real-world fall detection accuracy.

---

## Step 7: Verify Installation

Run both verification checks below from the `fall-detection-system` root folder with `(venv)` activated.

### 1. Import Check

```cmd
python -c "import torch; import cv2; import mediapipe; print('All imports OK')"
```

Expected output:

```text
All imports OK
```

### 2. Model Self-Test

Run the LSTM model module directly to test model creation, parameter count, forward pass, probability output, and save/load:

```cmd
python -m src.model
```

Expected output ends with:

```text
[OK] All model tests passed!
```

### 3. Optional End-to-End Smoke Test

Run a quick 5-epoch synthetic training pass or launch the interactive CLI demo:

```cmd
python -m src.train --synthetic --epochs 5
python app/demo.py
```

---

## Running in VS Code

1. **Open the project folder:**
   - Open VS Code → **File → Open Folder...** → select `fall-detection-system`.
   - *(Make sure you open `fall-detection-system` itself, not a parent folder, so `python -m src.model` resolves `src` properly.)*
2. **Select the virtual environment Python interpreter:**
   - Press `Ctrl + Shift + P` to open the Command Palette.
   - Type **`Python: Select Interpreter`** and press `Enter`.
   - Choose **`Python 3.x.x ('venv': venv)`** (path ending in `.\venv\Scripts\python.exe`).
   - *(If you don't see the option, install Microsoft's official **Python** extension from the VS Code Extensions sidebar `Ctrl + Shift + X`.)*
3. **Open and activate the integrated terminal:**
   - Press `` Ctrl + ` `` (backtick) to open the terminal inside VS Code.
   - VS Code usually activates `(venv)` automatically once the interpreter is selected. If `(venv)` is not shown at the prompt, run:
     - Command Prompt: `venv\Scripts\activate`
     - PowerShell: `venv\Scripts\Activate.ps1`
4. **Run scripts:**
   - From the integrated terminal (recommended for `-m` module commands and CLI flags):
     ```cmd
     python -m src.model
     python -m src.train --synthetic --epochs 5
     python app/demo.py
     ```
   - Or open a script such as `app\demo.py` and click the **▶ Run Python File** button in the top-right corner of the editor.

---

## Troubleshooting

### `"pip is not recognized"` (or `"python is not recognized"`)
- **Cause:** Python was installed without adding it to the Windows `PATH`, or you haven't opened a new terminal since installing.
- **Fix:**
  1. Open a brand-new Command Prompt and test `py --version`.
  2. If `python` still isn't recognized, open Windows **Apps & Features** (or re-run the Python installer `.exe`), choose **Modify**, make sure **"Add Python to environment variables"** / **"Add python.exe to PATH"** is checked, finish, and open a new terminal.

### `"No module named torch"` (or `No module named 'cv2'` / `'mediapipe'`)
- **Cause:** Your virtual environment is not activated, or you installed packages globally / in a different Python environment.
- **Fix:**
  1. Look for `(venv)` at the start of your terminal prompt. If missing, run `venv\Scripts\activate` (cmd) or `venv\Scripts\Activate.ps1` (PowerShell).
  2. Run `where python` — verify `...\fall-detection-system\venv\Scripts\python.exe` is listed first.
  3. Re-run Step 4 and Step 5 inside the activated `venv`.

### `"CUDA not available"` (`torch.cuda.is_available()` prints `False`)
- **Cause:** Either the CPU-only build of PyTorch was installed, your machine doesn't have an NVIDIA GPU, or your NVIDIA driver is too old for the installed CUDA wheel.
- **Fix:**
  1. Run `nvidia-smi`. If it errors, install or update your NVIDIA graphics driver from [nvidia.com/drivers](https://www.nvidia.com/drivers).
  2. Check your installed PyTorch version:
     ```cmd
     python -c "import torch; print(torch.__version__)"
     ```
     If it ends in `+cpu` (or has no `+cuXXX` suffix), uninstall and reinstall the CUDA build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/):
     ```cmd
     pip uninstall torch torchvision torchaudio -y
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```

### MediaPipe Installation Errors (`pip install mediapipe` fails)
- **Cause:** Outdated `pip`, 32-bit Python instead of 64-bit, or missing prebuilt wheel for a brand-new Python version.
- **Fix:**
  1. Upgrade `pip` first and retry:
     ```cmd
     python -m pip install --upgrade pip
     pip install mediapipe
     ```
  2. Confirm you are running **64-bit Python**:
     ```cmd
     python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
     ```
     It must print `64 bit`. (MediaPipe does not support 32-bit Python.)
  3. If wheels are unavailable for your exact Python minor version, install Python 3.11 or 3.12 (64-bit) and recreate `venv`.

### `"DLL load failed"` (when importing `torch`, `cv2`, or `mediapipe`)
- **Cause:** Missing Microsoft Visual C++ runtime libraries required by native C/C++ extensions on Windows.
- **Fix:** Download and install the latest **Microsoft Visual C++ Redistributable (x64)** (`vc_redist.x64.exe`) from Microsoft's official page: [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe), reboot if prompted, and re-run Step 7.

### Permission Errors (`Access is denied` / `PermissionError`)
- **Cause:** Windows folder permissions (for example, if the project was placed inside `C:\Program Files\` or another protected directory) or an antivirus lock.
- **Fix:**
  1. Move the project folder to a user-owned path like `C:\Users\<YourName>\Projects\fall-detection-system`.
  2. Or right-click Command Prompt / PowerShell / VS Code → **Run as administrator** and retry the command.

### PowerShell Script Execution Error (`running scripts is disabled on this system`)
- **Cause:** Windows PowerShell blocks local `.ps1` activation scripts by default.
- **Fix:** Run this once in PowerShell:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
  Then run `venv\Scripts\Activate.ps1` again (or switch to Command Prompt and run `venv\Scripts\activate`).

### Port Conflicts
- **Not applicable:** This project runs locally in the terminal and OpenCV desktop windows — it does **not** start a web server or bind to network ports.
