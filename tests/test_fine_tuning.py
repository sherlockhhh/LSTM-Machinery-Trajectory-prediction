"""Regression tests for validation-only search and recoverable batch runs."""

import copy
import json
from pathlib import Path

import pytest
import torch


def _base(tiny_hdf5):
    return {
        "dataset_path": tiny_hdf5, "seed": 42, "history_length": 3,
        "prediction_horizon": 2, "hidden_dim": 8, "num_layers": 1,
        "dropout": 0.2, "learning_rate": 0.001, "batch_size": 4,
        "epochs": 1, "patience": 1, "device": "cpu", "num_workers": 0,
        "loss": "smooth_l1", "optimizer": "adamw", "weight_decay": 0.0001,
    }


def _search():
    return {
        "selection_metric": "validation_ade",
        "screening": {"max_trials": 2, "epochs": 1, "patience": 1, "seed": 7},
        "search_space": {
            "history_length": [3], "hidden_dim": [4, 8], "num_layers": [1],
            "learning_rate": [0.001], "batch_size": [4],
            "dropout": [0.2], "weight_decay": [0.0001],
        },
        "final": {"epochs": 1, "patience": 1},
    }


def test_search_actually_varies_dropout_and_weight_decay():
    from hyperparameter_tuning import _candidate_parameters

    space = _search()["search_space"]
    space.update(dropout=[0.1, 0.3], weight_decay=[0.0001, 0.001])
    candidates = _candidate_parameters(space)
    assert len(candidates) == 8
    assert {(c["hidden_dim"], c["dropout"], c["weight_decay"]) for c in candidates} == {
        (4, 0.1, 0.0001), (4, 0.1, 0.001), (4, 0.3, 0.0001), (4, 0.3, 0.001),
        (8, 0.1, 0.0001), (8, 0.1, 0.001), (8, 0.3, 0.0001), (8, 0.3, 0.001),
    }


def test_unknown_search_parameter_is_not_silently_ignored():
    from hyperparameter_tuning import _candidate_parameters

    space = _search()["search_space"]
    space["droput"] = [0.3]
    with pytest.raises(ValueError, match="Unknown"):
        _candidate_parameters(space)


def test_ade_selects_actual_checkpoint_and_screening_does_not_evaluate_test(
    tiny_hdf5, tmp_path, monkeypatch,
):
    import train_lstm

    # Synthetic validation results intentionally disagree with loss ranking.
    # Keep training and weight saving real to catch saving the wrong epoch.
    scores = iter([(0.1, 0.3), (0.2, 0.1), (0.05, 0.2)])
    states = []

    def evaluate(model, *args):
        loss, ade = next(scores)
        states.append(copy.deepcopy(model.state_dict()))
        return loss, {"ade": ade, "fde": ade * 2, "mse": loss, "mae": ade}

    monkeypatch.setattr(train_lstm, "_evaluate", evaluate)
    config = _base(tiny_hdf5)
    config.update(epochs=5, patience=1, selection_metric="validation_ade", evaluate_test=False)
    result = train_lstm.run_training(config, tmp_path)
    assert result["best_epoch"] == 2
    assert result["validation_metrics"]["ade"] == 0.1
    assert result["test_metrics"] is None
    assert len(states) == 3
    checkpoint = torch.load(tmp_path / "best_model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert all(torch.equal(checkpoint["model_state_dict"][k], v) for k, v in states[1].items())
    assert not (tmp_path / "trajectory_example.png").exists()
    assert json.loads((tmp_path / "resolved_config.json").read_text(encoding="utf-8"))["evaluate_test"] is False
    assert json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))["epochs"][-1]["epoch"] == 3


def test_interrupted_search_resumes_without_retraining_completed_runs(
    tiny_hdf5, tmp_path, monkeypatch,
):
    import hyperparameter_tuning as tuner

    base, search = _base(tiny_hdf5), _search()
    real_training = tuner.run_training

    def interrupt_second(config, output_dir, **kwargs):
        if Path(output_dir).name == "trial_002":
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "training_history.json").write_text(
                '{"epochs": [{"epoch": 1, "validation_ade": 0.000001}]}', encoding="utf-8",
            )
            raise KeyboardInterrupt("simulated user interruption")
        return real_training(config, output_dir, **kwargs)

    monkeypatch.setattr(tuner, "run_training", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        tuner.run_tuning(base, search, tmp_path)
    partial = json.loads((tmp_path / "tuning_results.json").read_text(encoding="utf-8"))
    assert partial["completed_trials"] == 1
    assert partial["status"] != "complete"
    first_metrics = tmp_path / "trials" / "trial_001" / "metrics.json"
    first_timestamp = first_metrics.stat().st_mtime_ns
    assert json.loads(first_metrics.read_text(encoding="utf-8"))["test_metrics"] is None

    monkeypatch.setattr(tuner, "run_training", real_training)
    summary = tuner.run_tuning(base, search, tmp_path, resume=True)
    assert first_metrics.stat().st_mtime_ns == first_timestamp
    assert summary["status"] == "complete"
    assert summary["completed_trials"] == 2
    assert all(trial["best_validation_ade"] > 0.000001 for trial in summary["trials"])
    assert summary["final_result"]["test_metrics"]["ade"] >= 0
    for filename in ("leaderboard.csv", "summary.md", "search_manifest.json", "best_config.json"):
        assert (tmp_path / filename).is_file()

    final_path = tmp_path / "best_model" / "metrics.json"
    final_timestamp = final_path.stat().st_mtime_ns
    tuner.run_tuning(base, search, tmp_path, resume=True)
    assert final_path.stat().st_mtime_ns == final_timestamp

    changed = copy.deepcopy(search)
    changed["search_space"]["weight_decay"] = [0.001]
    with pytest.raises(ValueError, match="manifest|configuration"):
        tuner.run_tuning(base, changed, tmp_path, resume=True)
    assert first_metrics.stat().st_mtime_ns == first_timestamp


def test_search_refuses_to_overwrite_unrelated_output(tiny_hdf5, tmp_path):
    from hyperparameter_tuning import run_tuning

    marker = tmp_path / "existing.txt"
    marker.write_text("keep me")
    with pytest.raises((ValueError, FileExistsError)):
        run_tuning(_base(tiny_hdf5), _search(), tmp_path)
    assert marker.read_text() == "keep me"
