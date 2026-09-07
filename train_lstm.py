"""Train a baseline LSTM to predict future end-effector Cartesian positions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import FEATURE_DIM, FEATURE_NAMES, PreparedSplits, prepare_splits
from metrics import compute_metrics
from model import LSTMTrajectoryPredictor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset: torch.utils.data.Dataset, config: dict[str, Any], shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    loss_total = 0.0
    sample_total = 0
    predicted_trajectories: list[np.ndarray] = []
    true_trajectories: list[np.ndarray] = []
    with torch.no_grad():
        for history, normalized_target, last_position, future_positions in loader:
            history = history.to(device, non_blocking=True)
            normalized_target = normalized_target.to(device, non_blocking=True)
            last_position = last_position.to(device, non_blocking=True)
            prediction_normalized = model(history)
            loss = criterion(prediction_normalized, normalized_target)
            batch_size = history.shape[0]
            loss_total += loss.item() * batch_size
            sample_total += batch_size

            prediction_delta = prediction_normalized * target_std + target_mean
            prediction_absolute = last_position.unsqueeze(1) + prediction_delta
            predicted_trajectories.append(prediction_absolute.cpu().numpy())
            true_trajectories.append(future_positions.numpy())
    if sample_total == 0:
        raise ValueError("Evaluation split contains no usable trajectory windows.")
    return (
        loss_total / sample_total,
        compute_metrics(np.concatenate(predicted_trajectories), np.concatenate(true_trajectories)),
    )


def _write_trajectory_plot(
    model: nn.Module,
    test_split: torch.utils.data.Dataset,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> None:
    if len(test_split) == 0:
        return
    history, _, last_position, target_positions = test_split[0]
    model.eval()
    with torch.no_grad():
        prediction_normalized = model(history.unsqueeze(0).to(device))
        prediction_delta = prediction_normalized * target_std + target_mean
        prediction_positions = last_position.to(device).view(1, 1, 3) + prediction_delta
    raw_history = test_split.raw_history_positions(0)
    predicted = prediction_positions.squeeze(0).cpu().numpy()
    target = target_positions.numpy()

    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*raw_history.T, marker="o", label="history", color="#4c78a8")
    axis.plot(*target.T, marker="o", label="actual future", color="#54a24b")
    axis.plot(*predicted.T, marker="x", label="predicted future", color="#e45756")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("End-effector trajectory prediction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def _build_training_components(
    model: nn.Module, config: dict[str, Any]
) -> tuple[nn.Module, torch.optim.Optimizer, ReduceLROnPlateau | None, dict[str, Any]]:
    """Create regularized training components from explicit, serializable settings."""
    loss_name = str(config.get("loss", "mse")).lower()
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name in {"smooth_l1", "huber"}:
        criterion = nn.SmoothL1Loss()
        loss_name = "smooth_l1"
    else:
        raise ValueError("loss must be either 'mse' or 'smooth_l1'.")

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_class = AdamW if optimizer_name == "adamw" else Adam if optimizer_name == "adam" else None
    if optimizer_class is None:
        raise ValueError("optimizer must be either 'adam' or 'adamw'.")
    optimizer = optimizer_class(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=weight_decay
    )

    scheduler_config = config.get("scheduler")
    scheduler: ReduceLROnPlateau | None = None
    if scheduler_config is not None:
        if not isinstance(scheduler_config, dict):
            raise ValueError("scheduler must be an object or null.")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config.get("min_lr", 0.0)),
        )
    training_settings = {
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "loss": loss_name,
        "gradient_clip_norm": config.get("gradient_clip_norm"),
        "scheduler": scheduler_config,
    }
    return criterion, optimizer, scheduler, training_settings


def run_training(
    config: dict[str, Any], output_dir: str | Path, *, prepared: PreparedSplits | None = None,
) -> dict[str, Any]:
    """Fit the LSTM and write a checkpoint, metrics, normalizer and trajectory plot."""
    required = {
        "dataset_path", "seed", "history_length", "prediction_horizon", "hidden_dim",
        "num_layers", "dropout", "learning_rate", "batch_size", "epochs", "patience",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(missing)}")
    if int(config["epochs"]) < 1 or int(config["patience"]) < 1:
        raise ValueError("epochs and patience must both be positive.")
    selection_metric = str(config.get("selection_metric", "validation_normalized_loss"))
    if selection_metric not in {"validation_normalized_loss", "validation_ade"}:
        raise ValueError("selection_metric must be validation_normalized_loss or validation_ade.")
    evaluate_test = bool(config.get("evaluate_test", True))
    started = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _set_seed(int(config["seed"]))

    requested_device = str(config.get("device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        ("cpu" if requested_device == "auto" else requested_device)
    )
    if prepared is None:
        print(f"[{output_path.name}] Loading dataset on {device}...", flush=True)
        prepared = prepare_splits(
            config["dataset_path"],
            history_length=int(config["history_length"]),
            prediction_horizon=int(config["prediction_horizon"]),
            seed=int(config["seed"]),
        )
    if (prepared.train.history_length != int(config["history_length"])
            or prepared.train.prediction_horizon != int(config["prediction_horizon"])):
        raise ValueError("Cached dataset windows do not match the training configuration.")
    if min(len(prepared.train), len(prepared.valid), len(prepared.test)) == 0:
        raise ValueError("One split has no usable windows. Shorten the windows or use more data.")
    print(f"[{output_path.name}] device={device}, train_windows={len(prepared.train)}, "
          f"validation_windows={len(prepared.valid)}, selection={selection_metric}", flush=True)
    train_loader = _make_loader(prepared.train, config, shuffle=True, device=device)
    valid_loader = _make_loader(prepared.valid, config, shuffle=False, device=device)
    _write_json(output_path / "resolved_config.json", config)
    _write_json(output_path / "split_ids.json", {
        "train": list(prepared.train_demo_ids), "validation": list(prepared.valid_demo_ids),
        "test": list(prepared.test_demo_ids),
    })

    model = LSTMTrajectoryPredictor(
        input_dim=FEATURE_DIM,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        horizon=int(config["prediction_horizon"]),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion, optimizer, scheduler, training_settings = _build_training_components(model, config)
    target_mean = torch.tensor(prepared.normalizer.target_mean, device=device).view(1, 1, 3)
    target_std = torch.tensor(prepared.normalizer.target_std, device=device).view(1, 1, 3)

    best_validation_loss = float("inf")
    best_score = float("inf")
    best_validation_metrics: dict[str, float] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_path / "best_model.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        training_loss_total = 0.0
        training_sample_total = 0
        for batch_index, (inputs, normalized_target, _, _) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            normalized_target = normalized_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, normalized_target)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}.")
            loss.backward()
            gradient_clip_norm = config.get("gradient_clip_norm")
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            optimizer.step()
            training_loss_total += loss.item() * inputs.shape[0]
            training_sample_total += inputs.shape[0]
            progress_every = int(config.get("progress_every_batches", 0))
            if progress_every > 0 and batch_index % progress_every == 0:
                print(f"[{output_path.name}] epoch {epoch}/{config['epochs']} "
                      f"batch {batch_index}/{len(train_loader)} "
                      f"train_loss={training_loss_total / training_sample_total:.6f}", flush=True)

        validation_loss, validation_metrics = _evaluate(
            model, valid_loader, criterion, target_mean, target_std, device
        )
        if not all(math.isfinite(v) for v in [validation_loss, *validation_metrics.values()]):
            raise ValueError(f"Non-finite validation metric at epoch {epoch}.")
        score = validation_metrics["ade"] if selection_metric == "validation_ade" else validation_loss
        history.append({
            "epoch": epoch,
            "train_normalized_loss": training_loss_total / training_sample_total,
            "validation_normalized_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if scheduler is not None:
            scheduler.step(score)
        if score < best_score:
            best_score = score
            best_validation_loss = validation_loss
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "model_config": {
                    "input_dim": FEATURE_DIM,
                    "hidden_dim": int(config["hidden_dim"]),
                    "num_layers": int(config["num_layers"]),
                    "horizon": int(config["prediction_horizon"]),
                    "dropout": float(config["dropout"]),
                },
                "normalizer": prepared.normalizer.to_dict(),
                "feature_names": list(FEATURE_NAMES),
                "training": training_settings,
                "epoch": epoch,
                "selection_metric": selection_metric,
                "selection_value": score,
                "config": config,
            }, checkpoint_path.with_suffix(".pt.tmp"))
            checkpoint_path.with_suffix(".pt.tmp").replace(checkpoint_path)
        else:
            epochs_without_improvement += 1
        _write_json(output_path / "training_history.json", {"epochs": history})
        print(f"[{output_path.name}] epoch {epoch}/{config['epochs']} "
              f"train={history[-1]['train_normalized_loss']:.6f} val_loss={validation_loss:.6f} "
              f"val_ADE={validation_metrics['ade']:.6f}m val_FDE={validation_metrics['fde']:.6f}m "
              f"lr={history[-1]['learning_rate']:.2g} best_epoch={best_epoch} "
              f"time={history[-1]['epoch_seconds']:.1f}s", flush=True)
        if epochs_without_improvement >= int(config["patience"]):
            print(f"[{output_path.name}] Early stopping at epoch {epoch}.", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = None
    if evaluate_test:
        test_loader = _make_loader(prepared.test, config, shuffle=False, device=device)
        _, test_metrics = _evaluate(model, test_loader, criterion, target_mean, target_std, device)
        _write_trajectory_plot(
            model, prepared.test, target_mean, target_std, device, output_path / "trajectory_example.png"
        )
    _write_json(output_path / "normalizer.json", prepared.normalizer.to_dict())
    _write_json(output_path / "training_history.json", {"epochs": history})
    result = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_normalized_loss": best_validation_loss,
        "selection_metric": selection_metric,
        "selection_value": best_score,
        "validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "epochs_run": len(history),
        "training_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
        "dataset": {
            "train_windows": len(prepared.train),
            "validation_windows": len(prepared.valid),
            "test_windows": len(prepared.test),
            "train_demonstrations": len(prepared.train_demo_ids),
            "validation_demonstrations": len(prepared.valid_demo_ids),
            "test_demonstrations": len(prepared.test_demo_ids),
        },
        "training": training_settings,
    }
    _write_json(output_path / "metrics.json", result)
    return result


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        config["dataset_path"] = str((config_path.parent / dataset_path).resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, help="Optional override for a shorter verification run.")
    arguments = parser.parse_args()
    config = _load_config(arguments.config.resolve())
    if arguments.epochs is not None:
        config["epochs"] = arguments.epochs
    result = run_training(config, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
