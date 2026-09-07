# RoboMimic LSTM Trajectory Baseline Design

## Goal

Create one complete, reproducible LSTM baseline that predicts a Franka Panda end-effector's future Cartesian trajectory from a fixed history of RoboMimic observations. This baseline is for validating the complete data-to-metrics workflow; it deliberately does not perform hyperparameter search.

## Dataset

Input file: `robomimic_datasets/low_dim_dense_v15.hdf5`.

The inspected HDF5 file is the RoboMimic Can MG dense dataset. It contains 3,900 demonstrations, 150 time steps per demonstration, and a 20 Hz control frequency. Every frame exposes:

- `robot0_eef_pos`: 3-D end-effector position.
- `robot0_eef_quat`: 4-D end-effector orientation quaternion.
- `robot0_joint_pos`: 7-D joint position.
- `robot0_joint_vel`: 7-D joint velocity.
- `actions`: 7-D normalized control command.
- `object`: 14-D object/scene state.

`obs` values are used as history inputs. Future `robot0_eef_pos` values are the prediction target.

## Fixed Experimental Protocol

- Seed: 42.
- Split demonstrations, never individual time steps: 80% training, 10% validation, 10% test.
- History length: 12 frames (0.6 s).
- Prediction horizon: 12 frames (0.6 s).
- Input dimensions: 42 (`3 + 4 + 7 + 7 + 7 + 14`).
- Target: future 3-D end-effector displacement relative to the last history-frame position, shape `[12, 3]`.
- Model: two-layer LSTM with hidden size 128, dropout 0.2, followed by a linear output layer.
- Optimizer: Adam, learning rate `1e-3`, batch size 64, maximum 40 epochs, early stopping patience 8.
- Training loss: MSE over normalized target displacements.
- Device: CUDA when available, CPU otherwise.

Quaternion is an input feature only in this baseline. It will not be a prediction target because quaternion sign ambiguity would make a plain MSE trajectory metric misleading.

## Components

`dataset.py`

- Load only the required fields from HDF5.
- Split whole demonstration identifiers reproducibly.
- Fit input and target normalization statistics on training demonstrations only.
- Build legal `(history, future)` windows within each demonstration.
- Return PyTorch tensors and the last absolute end-effector position needed to reconstruct predictions.

`model.py`

- Define `LSTMTrajectoryPredictor` with a batch-first LSTM and a fully connected trajectory head.

`metrics.py`

- Compute MSE and MAE in Cartesian coordinate units.
- Compute ADE as average Euclidean distance over a horizon.
- Compute FDE as Euclidean distance at the final predicted time step.

`train_lstm.py`

- Load the fixed JSON configuration.
- Train, validate, save the best validation-ADE checkpoint, and run test evaluation once.
- Save training history, final metrics, feature metadata, normalization statistics, and an example trajectory plot under `outputs/`.

`config.json`

- Holds fixed baseline settings only. It is editable for later manual experiments but no parameter search loop is implemented.

`tests/`

- Use a generated, tiny HDF5 fixture to verify data window shapes and trajectory-level separation.
- Verify trajectory metrics on known coordinates.
- Verify the model output has `[batch, horizon, 3]` shape.

## Output Artifacts

The training run creates:

- `outputs/best_model.pt`: checkpoint including model state, normalization statistics, and configuration.
- `outputs/metrics.json`: validation and held-out test MSE, MAE, ADE, and FDE.
- `outputs/training_history.json`: train/validation loss and validation ADE for each epoch.
- `outputs/training_curve.png`: loss and ADE curves.
- `outputs/prediction_examples.png`: past trajectory, true future trajectory, and predicted future trajectory.

## Error Handling

- Fail with an explicit message when the HDF5 file is missing or required fields are absent.
- Reject a history length and horizon that leave no legal window.
- Protect standardization from zero-variance input dimensions.
- Keep all HDF5 access in the main process (`num_workers=0`) for reliable Windows execution.

## Out of Scope

- Hyperparameter search or parameter-combination comparison.
- Image, depth, force, or torque processing.
- Future end-effector orientation prediction.
- Online robot control or deployment.
