"""Two-stage, validation-driven hyperparameter tuning for the LSTM baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset import prepare_splits
from train_lstm import _load_config, _write_json, run_training


TUNABLE_PARAMETERS = (
    "history_length",
    "hidden_dim",
    "num_layers",
    "learning_rate",
    "batch_size",
)
OPTIONAL_TUNABLE_PARAMETERS = ("dropout", "weight_decay")


def _candidate_parameters(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    missing = [parameter for parameter in TUNABLE_PARAMETERS if parameter not in search_space]
    if missing:
        raise ValueError(f"Search space is missing parameters: {', '.join(missing)}")
    unknown = set(search_space).difference(TUNABLE_PARAMETERS + OPTIONAL_TUNABLE_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown search parameters: {', '.join(sorted(unknown))}")
    parameters = TUNABLE_PARAMETERS + tuple(
        name for name in OPTIONAL_TUNABLE_PARAMETERS if name in search_space
    )
    values = [search_space[parameter] for parameter in parameters]
    if any(not isinstance(parameter_values, list) or not parameter_values for parameter_values in values):
        raise ValueError("Every tuning parameter must map to a non-empty JSON list.")
    for name, entries in zip(parameters, values):
        for value in entries:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Invalid numerical value for {name}: {value!r}")
            if name in {"history_length", "hidden_dim", "num_layers", "batch_size"}:
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must contain positive integers.")
            elif name == "dropout" and not 0 <= value < 1:
                raise ValueError("dropout must be in [0, 1).")
            elif name == "learning_rate" and value <= 0:
                raise ValueError("learning_rate must be positive.")
            elif name == "weight_decay" and value < 0:
                raise ValueError("weight_decay must be nonnegative.")
        if len(set(entries)) != len(entries):
            raise ValueError(f"Duplicate values in search parameter {name}.")
    return [
        dict(zip(parameters, combination))
        for combination in itertools.product(*values)
    ]


def _sample_candidates(candidates: list[dict[str, Any]], max_trials: int, seed: int) -> list[dict[str, Any]]:
    if max_trials < 1:
        raise ValueError("screening.max_trials must be positive.")
    if max_trials >= len(candidates):
        return candidates
    selected_indices = np.random.default_rng(seed).choice(len(candidates), size=max_trials, replace=False)
    return [candidates[int(index)] for index in selected_indices]


def _trial_record(trial_number: int, parameters: dict[str, Any], trial_path: Path) -> dict[str, Any]:
    """Read the candidate's best physical-space validation trajectory error."""
    history_path = trial_path / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Completed trial is missing its history: {history_path}")
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)["epochs"]
    if not history:
        raise ValueError(f"Completed trial has no recorded epochs: {trial_path}")
    best_epoch = min(history, key=lambda epoch: epoch["validation_ade"])
    with (trial_path / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics["best_epoch"] != best_epoch["epoch"]:
        raise ValueError(f"Checkpoint epoch and validation ADE ranking disagree: {trial_path}")
    return {
        "trial": trial_number,
        "parameters": parameters,
        "best_validation_ade": best_epoch["validation_ade"],
        "best_validation_fde": best_epoch["validation_fde"],
        "best_validation_mse": best_epoch["validation_mse"],
        "best_epoch": best_epoch["epoch"],
        "epochs_run": len(history),
        "training_seconds": metrics.get("training_seconds"),
        "parameter_count": metrics.get("parameter_count"),
    }


def _manifest(base_config: dict[str, Any], tuning_config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(base_config["dataset_path"]).resolve()
    stat = dataset_path.stat()
    sources = ("dataset.py", "model.py", "metrics.py", "train_lstm.py", "hyperparameter_tuning.py")
    return {
        "schema_version": 1,
        "base_config": base_config,
        "tuning_config": tuning_config,
        "dataset": {"path": str(dataset_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in sources
        },
    }


def _completed_result(path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    # Only metrics.json marks successful completion; history alone can be partial.
    required = ("metrics.json", "best_model.pt", "training_history.json", "resolved_config.json",
                "normalizer.json", "split_ids.json")
    if not all((path / name).is_file() for name in required):
        return None
    with (path / "resolved_config.json").open(encoding="utf-8") as handle:
        saved_config = json.load(handle)
    with (path / "metrics.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    if saved_config != config or result.get("config") != config:
        raise ValueError(f"Completed run configuration differs: {path}. Use a new output directory.")
    return result


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    _write_json(path / "tuning_results.json", summary)
    fields = ["rank", "trial", *TUNABLE_PARAMETERS, *OPTIONAL_TUNABLE_PARAMETERS,
              "best_validation_ade", "best_validation_fde", "best_validation_mse",
              "best_epoch", "epochs_run", "training_seconds", "parameter_count"]
    csv_path = path / "leaderboard.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, trial in enumerate(summary["trials"], start=1):
            row = {key: value for key, value in trial.items() if key != "parameters"}
            row.update(trial["parameters"], rank=rank)
            writer.writerow(row)
    temporary.replace(csv_path)
    lines = [
        "# LSTM 精调结果", "",
        f"状态：{summary['status']}；已完成 {summary['completed_trials']}/{summary['screening']['max_trials']} 组。",
        "", "候选排序、早停及权重保存均以验证集 ADE 为准；筛选阶段不计算测试集指标。",
        "ADE / FDE 单位为米，MSE 单位为平方米。", "",
        "| 排名 | Trial | 隐藏维度 | 学习率 | Dropout | 权重衰减 | 验证 ADE | 验证 FDE | 最佳轮次 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, trial in enumerate(summary["trials"], start=1):
        p = trial["parameters"]
        lines.append(f"| {rank} | {trial['trial']} | {p['hidden_dim']} | {p['learning_rate']} | "
                     f"{p.get('dropout', 'base')} | {p.get('weight_decay', 'base')} | "
                     f"{trial['best_validation_ade']:.6f} | {trial['best_validation_fde']:.6f} | "
                     f"{trial['best_epoch']} |")
    reference = summary.get("reference")
    if reference and summary["best_trial"]:
        score = summary["best_trial"]["best_validation_ade"]
        gain = 100 * (1 - score / reference["validation_ade"])
        lines += ["", f"相对历史参考验证 ADE {reference['validation_ade']:.6f} m，"
                  f"本轮筛选最优降低 {gain:.2f}%（负数表示变差；单次实验比较）。"]
    final = summary.get("final_result")
    if final:
        test = final["test_metrics"]
        lines += ["", "## 最优配置正式复训", "",
                  f"最佳轮次：{final['best_epoch']}；验证 ADE：{final['validation_metrics']['ade']:.6f} m。",
                  f"测试 ADE：{test['ade']:.6f} m；测试 FDE：{test['fde']:.6f} m。"]
        if reference and reference.get("test_ade", 0) > 0:
            gain = 100 * (1 - test["ade"] / reference["test_ade"])
            lines.append(f"相对历史参考测试 ADE 降低 {gain:.2f}%（仅报告，不用于选参数）。")
    if summary.get("error"):
        lines += ["", f"停止原因：{summary['error']}", "使用同一命令加 --resume 继续。"]
    lines += ["", "续跑会跳过已完成候选和已完成复训；中断的那一组从第 1 轮重新训练。", ""]
    report_path = path / "summary.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)


def run_tuning(
    base_config: dict[str, Any], tuning_config: dict[str, Any], output_dir: str | Path, resume: bool = False
) -> dict[str, Any]:
    """Screen candidates on validation ADE, then retrain and test the winner.

    Candidate selection uses validation ADE in metres, so the score remains
    comparable when a candidate uses a different history-window length.
    Test-set metrics are not read or used for selection.
    """
    selection_metric = str(tuning_config.get("selection_metric", "validation_ade"))
    if selection_metric != "validation_ade":
        raise ValueError("This baseline tuner supports selection_metric='validation_ade'.")
    screening = tuning_config["screening"]
    final = tuning_config["final"]
    for stage in (screening, final):
        if int(stage["epochs"]) < 1 or int(stage["patience"]) < 1:
            raise ValueError("epochs and patience must both be positive.")
    output_path = Path(output_dir)
    trials_path = output_path / "trials"
    candidates = _sample_candidates(
        _candidate_parameters(tuning_config["search_space"]),
        int(screening["max_trials"]),
        int(screening["seed"]),
    )

    manifest = _manifest(base_config, tuning_config)
    manifest_path = output_path / "search_manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError("Output already contains a search. Use --resume or a new output directory.")
        with manifest_path.open(encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise ValueError("Search manifest differs (configuration, source code or dataset changed). "
                                 "Restore the original settings or choose a new output directory.")
    elif output_path.exists() and any(
        item.resolve() != Path(base_config["dataset_path"]).resolve() for item in output_path.iterdir()
    ):
        raise ValueError("Non-empty output has no compatible search manifest. Choose a new output directory.")
    trials_path.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    summary = {
        "status": "screening",
        "completed_trials": 0,
        "selection_metric": selection_metric,
        "selection_note": "Only validation ADE ranks candidates; test metrics do not influence selection.",
        "screening": {
            "max_trials": len(candidates),
            "epochs": int(screening["epochs"]),
            "patience": int(screening["patience"]),
            "seed": int(screening["seed"]),
        },
        "trials": [],
        "best_trial": None,
        "final_result": None,
        "reference": tuning_config.get("reference"),
    }
    # Cache only the latest window layout. Fine tuning uses a fixed layout,
    # so its 24 candidates and final run share one dataset preparation.
    cached_key = None
    cached_splits = None

    def train(config: dict[str, Any], path: Path) -> dict[str, Any]:
        nonlocal cached_key, cached_splits
        existing = _completed_result(path, config) if resume else None
        if existing is not None:
            print(f"[{path.name}] Reusing completed run.", flush=True)
            return existing
        key = (config["dataset_path"], config["seed"], config["history_length"], config["prediction_horizon"])
        if cached_key != key:
            print(f"[{path.name}] Preparing data: history={key[2]}, horizon={key[3]}...", flush=True)
            cached_splits = None
            cached_splits = prepare_splits(key[0], key[2], key[3], key[1])
            cached_key = key
        return run_training(config, path, prepared=cached_splits)

    try:
        _write_summary(output_path, summary)
        for trial_number, parameters in enumerate(candidates, start=1):
            trial_path = trials_path / f"trial_{trial_number:03d}"
            print(f"\nTrial {trial_number}/{len(candidates)}: "
                  f"{json.dumps(parameters, ensure_ascii=False)}", flush=True)
            trial_config = dict(base_config)
            trial_config.update(parameters)
            trial_config.update(epochs=int(screening["epochs"]), patience=int(screening["patience"]),
                                selection_metric=selection_metric, evaluate_test=False)
            train(trial_config, trial_path)
            summary["trials"].append(_trial_record(trial_number, parameters, trial_path))
            summary["trials"].sort(key=lambda trial: trial["best_validation_ade"])
            summary["completed_trials"] = len(summary["trials"])
            summary["best_trial"] = summary["trials"][0]
            _write_summary(output_path, summary)

        final_config = dict(base_config)
        final_config.update(summary["best_trial"]["parameters"])
        final_config.update(epochs=int(final["epochs"]), patience=int(final["patience"]),
                            selection_metric=selection_metric, evaluate_test=True)
        _write_json(output_path / "best_config.json", final_config)
        summary["status"] = "final_training"
        _write_summary(output_path, summary)
        print("\nScreening complete. Training selected configuration, then evaluating test set.", flush=True)
        summary["final_result"] = train(final_config, output_path / "best_model")
        summary["status"] = "complete"
        _write_summary(output_path, summary)
    except (KeyboardInterrupt, Exception) as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        _write_summary(output_path, summary)
        raise
    return summary


def _load_tuning_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-config", type=Path, default=Path("tuning_config.json"))
    parser.add_argument("--output-dir", type=Path, help="Override output_dir in the tuning configuration.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed candidate trial directories.")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without reading data or training.")
    arguments = parser.parse_args()
    tuning_config_path = arguments.tuning_config.resolve()
    tuning_config = _load_tuning_config(tuning_config_path)
    base_config_path = Path(tuning_config["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = tuning_config_path.parent / base_config_path
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = Path(tuning_config.get("output_dir", "outputs_tuning"))
        if not output_dir.is_absolute():
            output_dir = tuning_config_path.parent / output_dir
    base_config = _load_config(base_config_path.resolve())
    if arguments.dry_run:
        candidates = _sample_candidates(_candidate_parameters(tuning_config["search_space"]),
                                        int(tuning_config["screening"]["max_trials"]),
                                        int(tuning_config["screening"]["seed"]))
        print(json.dumps({"base_config": base_config, "output_dir": str(output_dir.resolve()),
                          "candidate_count": len(candidates), "candidates": candidates,
                          "screening": tuning_config["screening"], "final": tuning_config["final"]},
                         ensure_ascii=False, indent=2))
        return
    try:
        summary = run_tuning(base_config, tuning_config, output_dir, resume=arguments.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed trials are saved; rerun with --resume.", flush=True)
        raise SystemExit(130)
    print(json.dumps({"status": summary["status"], "best_trial": summary["best_trial"],
                      "test_metrics": summary["final_result"]["test_metrics"],
                      "summary_file": str(output_dir.resolve() / "summary.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
