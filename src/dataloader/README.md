# Dataloader Module

This module handles downloading, parsing, and loading the UCI Human Activity Recognition (HAR) dataset for motion prediction experiments.

## Overview

The dataloader provides a clean interface to work with raw inertial sensor time-series data from smartphones. It automates dataset acquisition and provides utilities for data inspection and validation.

### Key Components

- **`dataset.py`**: Core dataset loading functionality
- **`get_data.py`**: Automated dataset download and extraction from Google Drive
- **`quickcheck.py`**: Visualization tool for data sanity checks
- **`main.py`**: Entry point for running quickchecks

---

## The UCI HAR Dataset

### What It Contains

The UCI HAR dataset contains **inertial sensor readings** (accelerometer and gyroscope) recorded from smartphones carried by 30 subjects performing 6 different activities:

1. **WALKING**
2. **WALKING_UPSTAIRS**
3. **WALKING_DOWNSTAIRS**
4. **SITTING**
5. **STANDING**
6. **LAYING**

### Data Format

- **Sampling rate**: 50 Hz
- **Window size**: 128 samples per window (≈ 2.56 seconds)
- **Channels per window**: 6 (3 accelerometer + 3 gyroscope axes)
  - `body_acc_x`, `body_acc_y`, `body_acc_z`
  - `body_gyro_x`, `body_gyro_y`, `body_gyro_z`

### Train/Test Split

- **Training set**: ~7,350 windows
- **Test set**: ~2,950 windows
- **30 subjects**: Labeled in the dataset for cross-validation studies

---

## Module Components

### 1. `dataset.py` – UCIHARDatasetLoader

The main class for loading the dataset. Designed to work with **raw inertial signals**, not the engineered 561-feature vectors that some UCI HAR workflows use.

#### Class: `UCIHARDatasetLoader`

**Purpose**: Loads train/test splits of inertial time-series data.

**Constructor**:
```python
loader = UCIHARDatasetLoader(config_path="configs/config.yaml")
```

**Main Method**:
```python
split_data = loader.load_split(split="train")  # or "test"
```

**Returns**: `SplitData` namedtuple with:
- `X`: NumPy array of shape `(N, 128, 6)` – inertial signal windows
- `y`: NumPy array of shape `(N,)` – activity labels (1–6)
- `subject`: NumPy array of shape `(N,)` – subject IDs (1–30)

#### Dataclass: `SplitData`

Immutable container holding one train/test split:

```python
@dataclass(frozen=True)
class SplitData:
    X: np.ndarray          # (N, 128, 6) – inertial windows
    y: np.ndarray          # (N,) – activity labels
    subject: np.ndarray    # (N,) – subject IDs
```

#### Key Features

- **Configuration-driven**: Paths, channel names, and validation parameters come from `configs/config.yaml`
- **Channel flexibility**: Easily swap or reorder channels via config (uses aliases like "acc_x", "gyro_y", etc.)
- **Robust validation**: Checks array shapes, label ranges, and file existence
- **Clear error messages**: If data is missing or malformed, you get helpful diagnostics

#### Example Usage

```python
from src.dataloader.dataset import UCIHARDatasetLoader

# Initialize loader
loader = UCIHARDatasetLoader()

# Load training data
train_split = loader.load_split("train")
X_train = train_split.X        # (7346, 128, 6)
y_train = train_split.y        # (7346,)
subjects_train = train_split.subject  # (7346,)

# Load test data
test_split = loader.load_split("test")
X_test = test_split.X          # (2947, 128, 6)
y_test = test_split.y          # (2947,)

# Verify: print dataset description
print(loader.describe())
# Output: UCIHARDatasetLoader(dataset_root='data/raw/UCI_HAR_Dataset', 
#                              channels=[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
```

---

### 2. `get_data.py` – Dataset Download & Setup

Automated script to fetch the dataset from Google Drive, extract it, and place it in the correct project structure.

#### Main Function

```python
def main() -> None:
```

**Workflow** (7 steps):

1. **Download** from Google Drive using the file ID in `configs/config.yaml`
2. **Extract outer zip** to a temporary directory
3. **Detect nested zip** (if applicable) and extract
4. **Locate dataset root** by searching for the characteristic `train/`, `test/`, and `activity_labels.txt` files
5. **Move** to the final location (`data/raw/UCI_HAR_Dataset/`)
6. **Clean up** temporary files and the downloaded zip
7. **Sanity check** – verifies that expected files exist and have correct dimensions (128 columns)

#### When to Use

Run this once after cloning the repository (or after the dataset is deleted):

```bash
bash scripts/get_data.sh
```

Alternatively, from Python:

```python
from src.dataloader.get_data import main
main()
```

#### Configuration Dependency

The script reads these settings from `configs/config.yaml`:

```yaml
dataset:
  drive_file_id: "1pxWjPctWGAvybEKD556p6f_bqwXdTdhx"  # Google Drive file ID
  zip_name: "UCI_HAR_Dataset.zip"                       # Downloaded filename

paths:
  raw_data_dir: "data/raw"                              # Where zip goes
  dataset_dir: "data/raw/UCI_HAR_Dataset"               # Final location
  temp_dir: "data/raw/_tmp_uci_har"                     # Temporary extraction
```

---

### 3. `quickcheck.py` – Data Visualization & Validation

Generates a proof-of-life plot showing one sample window from each activity class, with all 6 channels plotted.

#### Class: `QuickChecker`

**Purpose**: Create a visual sanity check that the dataset loaded correctly.

**Constructor**:
```python
qc = QuickChecker(config_path="configs/config.yaml")
```

**Main Method**:
```python
result = qc.run()
```

**Returns**: `QuickcheckResult` with:
- `output_path`: Path to saved PNG plot
- `train_shape`: Shape of training data (should be `(7346, 128, 6)`)
- `label_counts`: Dictionary mapping class ID → count (sanity check)

#### Example Usage

```python
from src.dataloader.quickcheck import QuickChecker

qc = QuickChecker()
result = qc.run()

print(f"Plot saved to: {result.output_path}")
print(f"Train shape: {result.train_shape}")
print(f"Label distribution: {result.label_counts}")
```

#### Output

Generates `results/plots/sample_signals.png` containing:
- 6 rows (one per activity: WALKING, WALKING_UPSTAIRS, etc.)
- Each row plots all 6 inertial channels over the 128 time steps
- Helps visually confirm data quality and sensor behavior for each activity

#### Configuration

Settings in `configs/config.yaml`:

```yaml
quickcheck:
  activities:
    1: WALKING
    2: WALKING_UPSTAIRS
    # ... etc
  plot:
    figsize: [12, 14]
    dpi: 200
    filename: "sample_signals.png"
```

---

### 4. `main.py` – Entry Point

Simple entry point that runs the quickcheck:

```python
from quickcheck import QuickChecker

if __name__ == "__main__":
    qc = QuickChecker()
    qc.run()
```

**To run**:

```bash
cd src/dataloader
python main.py
```

---

## Typical Workflow

### Step 1: Download Dataset

```bash
# From repository root
bash scripts/get_data.sh
```

This downloads and extracts the UCI HAR dataset to `data/raw/UCI_HAR_Dataset/`.

### Step 2: Verify with Quickcheck

```bash
# From repository root or src/dataloader/
python -m src.dataloader.main
# Or: cd src/dataloader && python main.py
```

This creates a visual sanity check plot at `results/plots/sample_signals.png`.

### Step 3: Load Data in Your Experiments

```python
from src.dataloader.dataset import UCIHARDatasetLoader

loader = UCIHARDatasetLoader()
train_split = loader.load_split("train")
test_split = loader.load_split("test")

X_train, y_train, subj_train = train_split.X, train_split.y, train_split.subject
X_test, y_test, subj_test = test_split.X, test_split.y, test_split.subject

# Now use X_train, y_train, etc. in your models (RNN, EKF, PF)
```

---

## Configuration (`configs/config.yaml`)

The dataloader is controlled entirely by `configs/config.yaml`. Key sections:

```yaml
dataset:
  name: uci_har
  drive_file_id: "..."          # Google Drive ID for dataset zip
  zip_name: "UCI_HAR_Dataset.zip"
  channels:                      # Define sensor channels to load
    - [body_acc_x, acc_x]        # [filename_prefix, alias]
    - [body_acc_y, acc_y]
    - [body_acc_z, acc_z]
    - [body_gyro_x, gyro_x]
    - [body_gyro_y, gyro_y]
    - [body_gyro_z, gyro_z]
  window_length: 128             # Samples per window
  n_channels: 6                  # Number of channels
  label_range: [1, 6]            # Valid label range

paths:
  raw_data_dir: "data/raw"
  dataset_dir: "data/raw/UCI_HAR_Dataset"
  temp_dir: "data/raw/_tmp_uci_har"
  plot_output_dir: "results/plots"
```

You can modify channel order, sampling settings, or paths by editing this file.

---

## Common Issues & Troubleshooting

### Issue: `FileNotFoundError: Dataset root not found`

**Cause**: The dataset hasn't been downloaded yet.

**Solution**:
```bash
bash scripts/get_data.sh
```

### Issue: `ModuleNotFoundError: No module named 'yaml'`

**Cause**: Missing PyYAML dependency.

**Solution**:
```bash
pip install pyyaml
```

### Issue: Google Drive download fails

**Cause**: The file ID in config may be expired or the file sharing settings changed.

**Solution**: Check the config file and verify the Google Drive link is accessible.

### Issue: Sanity check complains about column count

**Cause**: The inertial signal files are corrupted or incomplete.

**Solution**: Re-download using `bash scripts/get_data.sh`.

---

## Design Decisions

1. **Raw Inertial Signals, Not Engineered Features**: This module loads the 128×6 inertial time-series windows, not the 561-feature engineered vectors that UCI HAR also provides. This is intentional—we want to study filtering and prediction on raw sensor data.

2. **Config-Driven**: All parameters (paths, channels, label range) come from `configs/config.yaml` rather than being hardcoded. This makes experiments reproducible and easy to modify.

3. **Immutable Data Classes**: `SplitData` is frozen to prevent accidental mutations during experiments.

4. **Separation of Concerns**:
   - `dataset.py`: Loading logic
   - `get_data.py`: Download & extraction
   - `quickcheck.py`: Visualization
   - `main.py`: CLI entry point

5. **Comprehensive Validation**: The loader validates array shapes, label ranges, and file existence, providing clear error messages if something is wrong.

---
