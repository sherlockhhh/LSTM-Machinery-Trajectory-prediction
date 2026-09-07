# RoboMimic LSTM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train one fixed-configuration LSTM that predicts the next 12 end-effector Cartesian positions from 12 history frames in the supplied RoboMimic Can MG dense dataset.

**Architecture:** Load full demonstrations and split them by identifier before constructing windows. Normalize 42-D history features and relative future Cartesian displacement with training-only statistics; the LSTM encodes each history and a linear head emits the full horizon. Training selects the lowest validation ADE checkpoint and evaluates it once on held-out demonstrations.

**Tech Stack:** Python 3, PyTorch, NumPy, h5py, Matplotlib, pytest.

**Spec:** `E:\\LSTM_trajectory_prediction\\docs\\superpowers\\specs\\2026-08-28-robomimic-lstm-baseline-design.md`

## Global Constraints

- Use `E:\\LSTM_trajectory_prediction\\robomimic_datasets\\low_dim_dense_v15.hdf5` as read-only input.
- Do not add a parameter-search loop.
- Use only trajectory-level train/validation/test splits with seed 42.
- Train on end-effector position, orientation, joint position, joint velocity, action, and object state; predict Cartesian position only.
- Keep HDF5 loading single-process for Windows compatibility.

---

### Task 1: Define the configuration and failing data/metric tests

**Files:**
- Create: `E:\\LSTM_trajectory_prediction\\config.json`
- Create: `E:\\LSTM_trajectory_prediction\\tests\\conftest.py`
- Create: `E:\\LSTM_trajectory_prediction\\tests\\test_dataset.py`
- Create: `E:\\LSTM_trajectory_prediction\\tests\\test_metrics.py`

**Interfaces:**
- Consumes: a test HDF5 file with `data/demo_i/obs/*` and `actions` datasets.
- Produces: expected public interfaces `RobomimicWindowDataset`, `prepare_splits`, `compute_metrics`.

- [ ] **Step 1: Write the failing data test**

```python
def test_prepared_windows_have_42_feature_channels(tiny_hdf5):
    from dataset import prepare_splits
    splits = prepare_splits(tiny_hdf5, history_length=3, prediction_horizon=2, seed=42)
    x, target_delta, last_pos, target_abs = splits.train[0]
    assert x.shape == (3, 42)
    assert target_delta.shape == (2, 3)
    assert last_pos.shape == (3,)
    assert target_abs.shape == (2, 3)
```

- [ ] **Step 2: Run the data test and verify it fails because `dataset` is absent**

Run: `pytest tests/test_dataset.py -v`

- [ ] **Step 3: Write the failing metric test**

```python
def test_metrics_are_zero_for_identical_trajectories():
    from metrics import compute_metrics
    pred = np.array([[[1., 2., 3.], [2., 2., 3.]]])
    result = compute_metrics(pred, pred.copy())
    assert result == {"mse": 0.0, "mae": 0.0, "ade": 0.0, "fde": 0.0}
```

- [ ] **Step 4: Run the metric test and verify it fails because `metrics` is absent**

Run: `pytest tests/test_metrics.py -v`

### Task 2: Implement verified dataset preparation

**Files:**
- Create: `E:\\LSTM_trajectory_prediction\\dataset.py`
- Modify: `E:\\LSTM_trajectory_prediction\\tests\\test_dataset.py`

**Interfaces:**
- Produces: `prepare_splits(hdf5_path: str, history_length: int, prediction_horizon: int, seed: int) -> PreparedSplits`.
- Produces: datasets whose items are `(history: Tensor[L,42], target_delta: Tensor[H,3], last_pos: Tensor[3], target_abs: Tensor[H,3])`.

- [ ] **Step 1: Implement whole-demo split and feature concatenation**

```python
FEATURE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_joint_vel", "object"]
features = np.concatenate([obs[key] for key in FEATURE_KEYS] + [actions], axis=-1).astype(np.float32)
```

Use a seeded permutation of sorted demo identifiers, assign 80/10/10, fit means and standard deviations only on concatenated training frames, and replace standard deviations smaller than `1e-8` with `1.0`.

- [ ] **Step 2: Build windows without crossing demonstration boundaries**

```python
for end_index in range(history_length - 1, length - prediction_horizon):
    history = normalized_features[end_index-history_length+1:end_index+1]
    last_pos = ee_pos[end_index]
    future_abs = ee_pos[end_index+1:end_index+1+prediction_horizon]
    target_delta = future_abs - last_pos
```

- [ ] **Step 3: Run all dataset tests**

Run: `pytest tests/test_dataset.py -v`

Expected: all dataset tests pass.

### Task 3: Implement the LSTM and metric modules with tests

**Files:**
- Create: `E:\\LSTM_trajectory_prediction\\model.py`
- Create: `E:\\LSTM_trajectory_prediction\\metrics.py`
- Create: `E:\\LSTM_trajectory_prediction\\tests\\test_model.py`

**Interfaces:**
- Produces: `LSTMTrajectoryPredictor(input_dim, hidden_dim, num_layers, horizon, dropout)`.
- Produces: `compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]`.

- [ ] **Step 1: Write the failing model-shape test**

```python
def test_lstm_returns_one_three_dimensional_path_per_sample():
    model = LSTMTrajectoryPredictor(42, 16, 2, 4, 0.2)
    assert model(torch.zeros(5, 3, 42)).shape == (5, 4, 3)
```

- [ ] **Step 2: Run the model test and verify it fails because `model` is absent**

Run: `pytest tests/test_model.py -v`

- [ ] **Step 3: Implement the model and metrics**

Use a batch-first LSTM, use the last layer's final hidden state, and map it through `nn.Linear(hidden_dim, horizon * 3)`. Compute MSE and MAE over coordinates; compute ADE from per-step Euclidean distance; compute FDE from final-step Euclidean distance.

- [ ] **Step 4: Run all unit tests**

Run: `pytest tests -v`

Expected: all data, metric, and model tests pass.

### Task 4: Implement training, artifacts, and an end-to-end smoke test

**Files:**
- Create: `E:\\LSTM_trajectory_prediction\\train_lstm.py`
- Create: `E:\\LSTM_trajectory_prediction\\requirements.txt`
- Create: `E:\\LSTM_trajectory_prediction\\README.md`

**Interfaces:**
- Consumes: configuration JSON and the supplied HDF5 path.
- Produces: `outputs/best_model.pt`, `outputs/metrics.json`, `outputs/training_history.json`, `outputs/training_curve.png`, and `outputs/prediction_examples.png`.

- [ ] **Step 1: Implement deterministic train/validation/test loops**

Set random seeds, use Adam and MSE loss on normalized target deltas, reconstruct absolute Cartesian positions before all reported metrics, select the checkpoint with minimum validation ADE, and stop after eight non-improving epochs.

- [ ] **Step 2: Implement output serialization and plots**

Write JSON with ordinary Python floats, save checkpoint state plus scalers/configuration, plot train/validation MSE and validation ADE, and plot past/true-future/predicted-future Cartesian paths for one held-out example.

- [ ] **Step 3: Run a one-epoch smoke test against the real file**

Run: `python train_lstm.py --config config.json --epochs 1 --output-dir outputs_smoke`

Expected: exit code 0 and all five output artifact types exist.

### Task 5: Run the fixed baseline experiment

**Files:**
- Create: `E:\\LSTM_trajectory_prediction\\outputs\\*`

**Interfaces:**
- Consumes: verified training program and full RoboMimic HDF5 file.
- Produces: final model and reported held-out metrics.

- [ ] **Step 1: Run the fixed 40-epoch baseline**

Run: `python train_lstm.py --config config.json --output-dir outputs`

- [ ] **Step 2: Verify the final artifacts and metrics**

Run: `Get-ChildItem outputs; Get-Content outputs\\metrics.json`

Expected: a checkpoint, two JSON reports, and two PNG plots; metrics include MSE, MAE, ADE, and FDE for the held-out test split.
