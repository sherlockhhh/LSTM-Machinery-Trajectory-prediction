"""Visualize good, typical, and difficult LSTM trajectory predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import PreparedSplits, prepare_splits
from model import LSTMTrajectoryPredictor


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        config["dataset_path"] = str((config_path.parent / dataset_path).resolve())
    return config


def _select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _predict_test_paths(
    config: dict[str, Any], checkpoint_path: Path
) -> tuple[PreparedSplits, np.ndarray, np.ndarray]:
    device = _select_device(config)
    prepared = prepare_splits(
        config["dataset_path"],
        history_length=int(config["history_length"]),
        prediction_horizon=int(config["prediction_horizon"]),
        seed=int(config["seed"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LSTMTrajectoryPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    target_mean = torch.tensor(prepared.normalizer.target_mean, device=device).view(1, 1, 3)
    target_std = torch.tensor(prepared.normalizer.target_std, device=device).view(1, 1, 3)

    predicted_paths: list[np.ndarray] = []
    actual_paths: list[np.ndarray] = []
    loader = DataLoader(prepared.test, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for history, _, last_position, future_positions in loader:
            prediction_normalized = model(history.to(device))
            prediction_delta = prediction_normalized * target_std + target_mean
            prediction_absolute = last_position.to(device).unsqueeze(1) + prediction_delta
            last_position_numpy = last_position.numpy()[:, None, :]
            predicted_paths.append(prediction_absolute.cpu().numpy() - last_position_numpy)
            actual_paths.append(future_positions.numpy() - last_position_numpy)
    if not predicted_paths:
        raise ValueError("The test split has no usable trajectory windows.")
    origin = np.zeros((len(prepared.test), 1, 3), dtype=np.float32)
    return (
        prepared,
        np.concatenate((origin, np.concatenate(actual_paths, axis=0)), axis=1),
        np.concatenate((origin, np.concatenate(predicted_paths, axis=0)), axis=1),
    )


def _representative_indices(actual_paths: np.ndarray, predicted_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ade = np.linalg.vector_norm(actual_paths[:, 1:] - predicted_paths[:, 1:], axis=-1).mean(axis=1)
    order = np.argsort(ade)
    percentiles = np.asarray((25, 50, 90), dtype=np.int32)
    rank_indices = np.floor((len(order) - 1) * percentiles / 100).astype(int)
    return percentiles, order[rank_indices]


def create_representative_trajectory_comparison(
    config: dict[str, Any], checkpoint_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Create one 3D comparison panel for each selected ADE percentile."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")
    prepared, actual_paths, predicted_paths = _predict_test_paths(config, checkpoint_path)
    if len(actual_paths) < 3:
        raise ValueError("At least three test windows are required for representative visualization.")

    percentiles, selected_indices = _representative_indices(actual_paths, predicted_paths)
    titles = {25: "Low-error example", 50: "Typical example", 90: "Challenging example"}
    selected_path_values = np.concatenate(
        (actual_paths[selected_indices], predicted_paths[selected_indices]), axis=0
    )
    axis_extent = max(float(np.max(np.abs(selected_path_values))) * 1.15, 0.03)
    figure = plt.figure(figsize=(16, 5.2))
    samples: list[dict[str, float | int]] = []
    for panel_index, (percentile, sample_index) in enumerate(zip(percentiles, selected_indices), start=1):
        axis = figure.add_subplot(1, 3, panel_index, projection="3d")
        history = prepared.test.raw_history_positions(int(sample_index))
        history_relative = history - history[-1]
        actual = actual_paths[sample_index]
        predicted = predicted_paths[sample_index]
        point_distances = np.linalg.vector_norm(actual[1:] - predicted[1:], axis=-1)
        ade = float(point_distances.mean())
        fde = float(point_distances[-1])

        axis.plot(*history_relative.T, "o-", color="#4c78a8", label="history")
        axis.plot(*actual.T, "o-", color="#54a24b", label="actual future")
        axis.plot(*predicted.T, "x-", color="#e45756", label="predicted future")
        axis.scatter(0, 0, 0, color="#202020", s=25, zorder=5)
        axis.set_xlim(-axis_extent, axis_extent)
        axis.set_ylim(-axis_extent, axis_extent)
        axis.set_zlim(-axis_extent, axis_extent)
        axis.set_xlabel("ΔX (m)")
        axis.set_ylabel("ΔY (m)")
        axis.set_zlabel("ΔZ (m)")
        axis.set_title(
            f"{titles[int(percentile)]} (ADE P{int(percentile)})\n"
            f"ADE={ade * 100:.2f} cm, FDE={fde * 100:.2f} cm"
        )
        if panel_index == 1:
            axis.legend(loc="upper left", fontsize=8)
        samples.append({
            "percentile": int(percentile),
            "index": int(sample_index),
            "ade_m": ade,
            "fde_m": fde,
        })

    figure.suptitle("Representative end-effector trajectory predictions", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"test_windows": int(len(actual_paths)), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("representative_trajectory_comparison.png"))
    arguments = parser.parse_args()
    summary = create_representative_trajectory_comparison(
        _load_config(arguments.config.resolve()), arguments.checkpoint, arguments.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
