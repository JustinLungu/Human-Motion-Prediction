# Data used for this project

## Engineered feature vectors (NOT USED)

Files:
- X_train.txt, X_test.txt
- features.txt
- features_info.txt

These are hand-engineered statistical and frequency-domain features computed over a 2.56 s window.

They are meant for activity classification, not state estimation.


We will ignore these for modeling.

## Raw inertial time series (USED)

```
train/Inertial Signals/
test/Inertial Signals/
```

Files (per split):
```
body_acc_[x,y,z]_*.txt
body_gyro_[x,y,z]_*.txt
total_acc_[x,y,z]_*.txt
```

Each file:
- shape (N, 128)
- sampled at 50 Hz
- one row = one time window
- window length = 128 samples (2.56 s)

Ideal for EKF, Particle Filter, and RNN.


## Understand Labels and Subjects

File ``` activiti_labels.txt``` contains 6 different types of motion.
Labels per sample: ```y_train.txt / y_test.txt ```. 