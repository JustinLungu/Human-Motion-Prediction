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

## Dataset
This project uses the **UCI Human Activity Recognition (HAR)** dataset.
Download link:  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Raw data should be placed in:
