"""Trajectory error metrics used by the LSTM baseline."""

from __future__ import annotations

import numpy as np


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute Cartesian MSE, MAE, ADE and FDE.

    Inputs must have shape ``[batch, horizon, 3]`` and be expressed in metres.
    ADE and FDE use Euclidean distance per trajectory point.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(f"Prediction shape {predictions.shape} does not match target shape {targets.shape}.")
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("Predictions and targets must have shape [batch, horizon, 3].")
    error = predictions - targets
    point_distances = np.linalg.norm(error, axis=-1)
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "ade": float(np.mean(point_distances)),
        "fde": float(np.mean(point_distances[:, -1])),
    }
