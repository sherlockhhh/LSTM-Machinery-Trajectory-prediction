from pathlib import Path
import warnings


def test_representative_comparison_selects_three_ade_percentiles(tiny_hdf5, tmp_path: Path):
    from train_lstm import run_training
    from visualize_representative_trajectories import create_representative_trajectory_comparison

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
    model_dir = tmp_path / "model"
    run_training(config, model_dir)
    output_path = tmp_path / "representative_trajectories.png"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        summary = create_representative_trajectory_comparison(
            config, model_dir / "best_model.pt", output_path
        )

    assert summary["test_windows"] == 4
    assert [sample["percentile"] for sample in summary["samples"]] == [25, 50, 90]
    assert len({sample["index"] for sample in summary["samples"]}) == 3
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
