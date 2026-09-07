from pathlib import Path


def test_tuner_selects_by_validation_metric_and_writes_reproducible_artifacts(tiny_hdf5, tmp_path: Path):
    from hyperparameter_tuning import run_tuning

    base_config = {
        "dataset_path": str(tiny_hdf5),
        "seed": 42,
        "history_length": 3,
        "prediction_horizon": 2,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "epochs": 1,
        "patience": 1,
        "num_workers": 0,
        "device": "cpu",
    }
    tuning_config = {
        "selection_metric": "validation_ade",
        "screening": {"max_trials": 2, "epochs": 1, "patience": 1, "seed": 7},
        "search_space": {
            "history_length": [3],
            "hidden_dim": [4, 8],
            "num_layers": [1],
            "learning_rate": [1e-3],
            "batch_size": [4],
        },
        "final": {"epochs": 1, "patience": 1},
    }

    summary = run_tuning(base_config, tuning_config, tmp_path)

    scores = [trial["best_validation_ade"] for trial in summary["trials"]]
    assert summary["selection_metric"] == "validation_ade"
    assert len(summary["trials"]) == 2
    assert scores == sorted(scores)
    assert (tmp_path / "tuning_results.json").is_file()
    assert (tmp_path / "best_config.json").is_file()
    assert (tmp_path / "best_model" / "best_model.pt").is_file()

    trial_metrics_path = tmp_path / "trials" / "trial_001" / "metrics.json"
    first_write_time = trial_metrics_path.stat().st_mtime_ns
    run_tuning(base_config, tuning_config, tmp_path, resume=True)
    assert trial_metrics_path.stat().st_mtime_ns == first_write_time
