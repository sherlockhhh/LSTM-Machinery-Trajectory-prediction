import numpy as np


def test_metrics_are_zero_for_identical_trajectories():
    from metrics import compute_metrics

    prediction = np.array([[[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]])
    result = compute_metrics(prediction, prediction.copy())

    assert result == {"mse": 0.0, "mae": 0.0, "ade": 0.0, "fde": 0.0}


def test_metrics_measure_final_displacement_error():
    from metrics import compute_metrics

    prediction = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
    target = np.zeros_like(prediction)
    result = compute_metrics(prediction, target)

    assert result["fde"] == 5.0
    assert result["ade"] == 2.5
