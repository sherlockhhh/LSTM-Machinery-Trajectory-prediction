from pathlib import Path


def test_training_writes_reusable_artifacts(tiny_hdf5, tmp_path: Path):
    from train_lstm import run_training

    config = {
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

    result = run_training(config, tmp_path)

    assert result["test_metrics"]["mse"] >= 0.0
    for filename in (
        "best_model.pt",
        "metrics.json",
        "normalizer.json",
        "training_history.json",
        "trajectory_example.png",
    ):
        assert (tmp_path / filename).is_file()


def test_training_records_regularized_training_settings(tiny_hdf5, tmp_path: Path):
    """The anti-overfitting training options must be applied and reproducible."""
    from train_lstm import run_training

    config = {
        "dataset_path": str(tiny_hdf5),
        "seed": 42,
        "history_length": 3,
        "prediction_horizon": 2,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "epochs": 1,
        "patience": 1,
        "num_workers": 0,
        "device": "cpu",
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "loss": "smooth_l1",
        "gradient_clip_norm": 1.0,
        "scheduler": {"factor": 0.5, "patience": 1, "min_lr": 1e-6},
    }

    result = run_training(config, tmp_path)

    assert result["training"]["optimizer"] == "adamw"
    assert result["training"]["loss"] == "smooth_l1"
    assert result["training"]["weight_decay"] == 1e-4
    assert result["training"]["gradient_clip_norm"] == 1.0
    assert result["training"]["scheduler"]["factor"] == 0.5

    history = (tmp_path / "training_history.json").read_text(encoding="utf-8")
    assert "train_normalized_loss" in history
    assert "validation_normalized_loss" in history
    assert "train_normalized_mse" not in history
