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

# Signal-Level Prediction Baselines

To move beyond activity-level classification and directly evaluate temporal modeling of IMU signals, the problem was reformulated as a one-step-ahead prediction task. Given a fixed-length window of past inertial measurements, the goal is to predict the next IMU sample. Performance is measured using Root Mean Squared Error (RMSE), averaged across channels.

Each original UCI HAR window contains 128 time steps. For prediction, the first 127 steps are used as input, and the final step is treated as the prediction target. This formulation is shared across all signal-level models to ensure fair comparison.

## Baseline: Persistence Model

As a minimal signal-level reference, a persistence baseline was implemented. This model predicts the next IMU sample by simply copying the last observed value in the input window, i.e., assuming no change in the signal dynamics.

This baseline captures short-term temporal continuity but does not model trends, noise characteristics, or longer-term dependencies.

### Results

- **Metric**: Mean RMSE (lower is better)
- **Train RMSE**: 0.1099
- **Test RMSE**: 0.1077

Per-channel errors show that some IMU axes are inherently more difficult to predict than others, with consistently higher error on certain gyroscope channels.

### Interpretation

The persistence baseline establishes a meaningful signal-level lower bound. Its similar performance on training and test data indicates stable behavior and confirms that the evaluation pipeline is correct. However, the relatively high RMSE demonstrates that simple temporal continuity is insufficient to accurately predict future IMU readings, motivating more expressive temporal models.


## Baseline: Recurrent Neural Network (RNN)

To provide a data-driven temporal modeling baseline, a lightweight recurrent neural network (LSTM) was implemented. The RNN processes the full input sequence and uses the final hidden state to predict the next IMU sample.

The model is intentionally simple, serving as a reference for learned temporal dependencies rather than a highly optimized neural architecture.

### Results

- **Metric**: Mean RMSE (lower is better)
- **Train RMSE**: 0.0645
- **Test RMSE**: 0.0652

The RNN substantially outperforms the persistence baseline on both training and test sets, with a consistent reduction in RMSE across all channels.

### Interpretation

The RNN’s improved performance demonstrates the benefit of learning temporal structure directly from data. The close match between training and test RMSE suggests good generalization and limited overfitting under the current setup. These results confirm that the signal contains exploitable temporal patterns beyond simple persistence.

The RNN baseline therefore serves as a strong data-driven reference point against which model-based approaches can be compared.