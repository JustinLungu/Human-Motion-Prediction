# Types of Models, Information, Conclusions

## Baseline: Majority Class Classifier

As an initial sanity check and lower-bound reference, a simple majority class baseline was implemented. This model ignores all inertial sensor signals and always predicts the most frequent activity label observed in the training set.

The purpose of this baseline is not to achieve competitive performance, but to validate the end-to-end experimental pipeline and establish a minimal performance threshold that any meaningful model must exceed.

### Results

- **Model**: Majority class baseline
- **Training samples**: 7,352 
- **Training accuracy**: 0.1914

Given that the UCI HAR dataset contains six activity classes, random guessing would yield an expected accuracy of approximately 0.167. The observed accuracy of 0.191 indicates a mild class imbalance, but confirms that label frequency alone provides only limited predictive power.


### Interpretation

This result demonstrates that:
- The dataset is not trivially solvable through class imbalance alone.
- Meaningful temporal or signal-based modeling is required to achieve substantial improvements.
- The data loading, model interface, and evaluation pipeline are functioning correctly.

This majority baseline therefore serves as a lower bound for subsequent models. All state-estimation and learning-based approaches (EKF, Particle Filter, RNN) are expected to significantly outperform this baseline.

The baseline is considered complete and will be used as a reference point for future experiments.

### Next Steps

While the majority class baseline provides a useful lower bound, it does not exploit the temporal structure or physical characteristics of inertial measurement unit (IMU) signals. In the next stage of this project, more informative baselines will be introduced that operate directly on the raw time-series data. These include a simple persistence model as a signal-level reference, as well as three principled approaches for temporal modeling: an Extended Kalman Filter (EKF), a Particle Filter (PF), and a lightweight recurrent neural network (RNN). These methods incorporate increasing levels of temporal awareness and modeling capacity, and their performance will be evaluated relative to the majority baseline to assess the benefits of dynamic state estimation and data-driven sequence modeling.


