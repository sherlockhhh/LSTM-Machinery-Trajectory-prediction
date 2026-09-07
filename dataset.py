"""robomimic low-dimensional trajectory dataset utilities.

This module deliberately reads only numerical robot-state fields. Images are
not needed for the baseline LSTM trajectory-prediction task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "actions",
    "object",
)
FEATURE_DIM = 42
TARGET_DIM = 3


@dataclass(frozen=True)
class Normalizer:
    """Statistics fitted with training trajectories only."""

    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


@dataclass
class _Trajectory:
    features: np.ndarray
    eef_pos: np.ndarray


class RobomimicWindowDataset(Dataset):
    """Sliding time windows from a trajectory-level split of robomimic data."""

    def __init__(
        self,
        trajectories: Sequence[_Trajectory],
        history_length: int,
        prediction_horizon: int,
        normalizer: Normalizer,
    ) -> None:
        self.trajectories = list(trajectories)
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.normalizer = normalizer
        self._window_index = self._build_window_index()

    def _build_window_index(self) -> np.ndarray:
        indices: list[tuple[int, int]] = []
        required_steps = self.history_length + self.prediction_horizon
        for trajectory_index, trajectory in enumerate(self.trajectories):
            if len(trajectory.eef_pos) < required_steps:
                continue
            first_last_history_index = self.history_length - 1
            last_last_history_index = len(trajectory.eef_pos) - self.prediction_horizon - 1
            indices.extend(
                (trajectory_index, last_history_index)
                for last_history_index in range(
                    first_last_history_index, last_last_history_index + 1
                )
            )
        return np.asarray(indices, dtype=np.int32).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectory_index, last_history_index = self._window_index[index]
        trajectory = self.trajectories[int(trajectory_index)]
        history_start = int(last_history_index) - self.history_length + 1
        history = trajectory.features[history_start : int(last_history_index) + 1]
        last_position = trajectory.eef_pos[int(last_history_index)]
        future_positions = trajectory.eef_pos[
            int(last_history_index) + 1 : int(last_history_index) + 1 + self.prediction_horizon
        ]
        target_delta = future_positions - last_position

        normalized_history = (history - self.normalizer.input_mean) / self.normalizer.input_std
        normalized_target = (target_delta - self.normalizer.target_mean) / self.normalizer.target_std
        return (
            torch.from_numpy(normalized_history.astype(np.float32, copy=False)),
            torch.from_numpy(normalized_target.astype(np.float32, copy=False)),
            torch.from_numpy(last_position.astype(np.float32, copy=False)),
            torch.from_numpy(future_positions.astype(np.float32, copy=False)),
        )

    def raw_history_positions(self, index: int) -> np.ndarray:
        """Return Cartesian positions in the historical window for plotting."""
        trajectory_index, last_history_index = self._window_index[index]
        trajectory = self.trajectories[int(trajectory_index)]
        start = int(last_history_index) - self.history_length + 1
        return trajectory.eef_pos[start : int(last_history_index) + 1].copy()


@dataclass(frozen=True)
class PreparedSplits:
    train: RobomimicWindowDataset
    valid: RobomimicWindowDataset
    test: RobomimicWindowDataset
    train_demo_ids: tuple[str, ...]
    valid_demo_ids: tuple[str, ...]
    test_demo_ids: tuple[str, ...]
    normalizer: Normalizer


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (0, name)


def _load_trajectory(demo_group: h5py.Group) -> _Trajectory:
    if "obs" not in demo_group or "actions" not in demo_group:
        raise ValueError("Each demo must contain an 'obs' group and an 'actions' dataset.")
    obs = demo_group["obs"]
    required_observations = FEATURE_NAMES[:-2] + ("object",)
    missing = [key for key in required_observations if key not in obs]
    if missing:
        raise ValueError(f"Demo is missing required observation fields: {', '.join(missing)}")

    parts = [np.asarray(obs[key], dtype=np.float32) for key in FEATURE_NAMES[:-2]]
    parts.append(np.asarray(demo_group["actions"], dtype=np.float32))
    parts.append(np.asarray(obs["object"], dtype=np.float32))
    lengths = {part.shape[0] for part in parts}
    if len(lengths) != 1:
        raise ValueError("Observation fields and actions must have the same number of time steps.")
    features = np.concatenate(parts, axis=1)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} input features, found {features.shape[1]}. "
            "This baseline targets robomimic low-dimensional datasets."
        )
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    return _Trajectory(features=features, eef_pos=eef_pos)


def _split_demo_ids(demo_ids: Sequence[str], seed: int) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(demo_ids) < 3:
        raise ValueError("At least three demonstrations are required for train/validation/test splits.")
    shuffled = np.asarray(demo_ids, dtype=object)[np.random.default_rng(seed).permutation(len(demo_ids))]
    train_end = max(1, int(len(shuffled) * 0.8))
    valid_end = max(train_end + 1, int(len(shuffled) * 0.9))
    valid_end = min(valid_end, len(shuffled) - 1)
    return tuple(shuffled[:train_end]), tuple(shuffled[train_end:valid_end]), tuple(shuffled[valid_end:])


def _fit_normalizer(
    trajectories: Sequence[_Trajectory], history_length: int, prediction_horizon: int
) -> Normalizer:
    feature_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    feature_sum_sq = np.zeros(FEATURE_DIM, dtype=np.float64)
    feature_count = 0
    target_sum = np.zeros(TARGET_DIM, dtype=np.float64)
    target_sum_sq = np.zeros(TARGET_DIM, dtype=np.float64)
    target_count = 0

    for trajectory in trajectories:
        features = trajectory.features.astype(np.float64, copy=False)
        feature_sum += features.sum(axis=0)
        feature_sum_sq += np.square(features).sum(axis=0)
        feature_count += len(features)

        first_last = history_length - 1
        last_last = len(trajectory.eef_pos) - prediction_horizon - 1
        for last_history_index in range(first_last, last_last + 1):
            deltas = trajectory.eef_pos[
                last_history_index + 1 : last_history_index + 1 + prediction_horizon
            ] - trajectory.eef_pos[last_history_index]
            target_sum += deltas.sum(axis=0)
            target_sum_sq += np.square(deltas).sum(axis=0)
            target_count += len(deltas)

    if feature_count == 0 or target_count == 0:
        raise ValueError("Training trajectories are too short for the requested history and prediction windows.")
    input_mean = feature_sum / feature_count
    input_var = np.maximum(feature_sum_sq / feature_count - np.square(input_mean), 1e-8)
    target_mean = target_sum / target_count
    target_var = np.maximum(target_sum_sq / target_count - np.square(target_mean), 1e-8)
    return Normalizer(
        input_mean=input_mean.astype(np.float32),
        input_std=np.sqrt(input_var).astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=np.sqrt(target_var).astype(np.float32),
    )


def prepare_splits(
    dataset_path: str | Path,
    history_length: int,
    prediction_horizon: int,
    seed: int = 42,
) -> PreparedSplits:
    """Read a robomimic HDF5 file and prepare leakage-free trajectory splits."""
    if history_length < 1 or prediction_horizon < 1:
        raise ValueError("history_length and prediction_horizon must both be positive.")
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    with h5py.File(path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise ValueError("Expected a robomimic HDF5 file with a top-level 'data' group.")
        data_group = hdf5_file["data"]
        demo_ids = sorted(data_group.keys(), key=_demo_sort_key)
        trajectories = {demo_id: _load_trajectory(data_group[demo_id]) for demo_id in demo_ids}

    train_ids, valid_ids, test_ids = _split_demo_ids(demo_ids, seed)
    normalizer = _fit_normalizer(
        [trajectories[demo_id] for demo_id in train_ids], history_length, prediction_horizon
    )
    return PreparedSplits(
        train=RobomimicWindowDataset(
            [trajectories[demo_id] for demo_id in train_ids], history_length, prediction_horizon, normalizer
        ),
        valid=RobomimicWindowDataset(
            [trajectories[demo_id] for demo_id in valid_ids], history_length, prediction_horizon, normalizer
        ),
        test=RobomimicWindowDataset(
            [trajectories[demo_id] for demo_id in test_ids], history_length, prediction_horizon, normalizer
        ),
        train_demo_ids=train_ids,
        valid_demo_ids=valid_ids,
        test_demo_ids=test_ids,
        normalizer=normalizer,
    )
