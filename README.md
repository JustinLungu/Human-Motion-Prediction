# Human Motion State Estimation Using EKF, Particle Filtering, and RNNs

This project performs one-step-ahead prediction and filtering of human IMU signals
using the UCI Human Activity Recognition (HAR) dataset. It compares three
estimation approaches:

- Extended Kalman Filter (EKF)
- Particle Filter (PF)
- A simple Recurrent Neural Network (RNN) baseline

The goal is to study:
1. How model-based estimators compare with a data-driven model.
2. How each method behaves under normal conditions.
3. How robustness changes when measurement quality worsens (noise, drift, dropouts).

## Features

- Clean EKF and PF implementations
- Lightweight PyTorch RNN baseline
- Configurable experiments and reproducible evaluation
- Robustness tests through synthetic noise injection and data degradation
- Clear visualizations (filter outputs, residuals, error curves)

---

## Environment Setup

This project uses a Python virtual environment to ensure isolation and
reproducibility.

### 1. Create the virtual environment

From the repository root:

```
bash scripts/create_venv.sh
```

This will create a virtual environment named ```uci_project``` and install all
required dependencies.

### 2. Activate the virtual environment

```
source uci_project/bin/activate
```

After activation, your terminal prompt should indicate that the environment is
active.


### Dataset

## Dataset

This project uses the **UCI Human Activity Recognition (HAR)** dataset.

Official dataset page:  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

### Download and prepare the dataset

The dataset download and extraction process is fully automated.

From the repository root, run:

```
bash scripts/get_data.sh
```

The dataset files are not committed to the repository and are generated locally
by the download script.


### Repository Structure

```
.
├── configs/        # Configuration files
├── data/
│   ├── raw/        # Raw dataset (auto-generated)
│   └── processed/ # Processed data
├── results/        # Metrics and plots
├── report/         # LaTeX report and figures
├── scripts/        # Setup and utility scripts
├── src/            # Source code (filters, models, experiments)
├── requirements.txt
└── README.md
```