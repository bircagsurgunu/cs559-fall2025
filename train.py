"""
Experiment orchestration script for CS559 Fall 2025 facial attractiveness regression.
"""
from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict, Tuple, List

import numpy as np
import tensorflow as tf

from model import (
    ModelConfig,
    build_model,
    compile_model,
    train_one_run,
    evaluate,
    rounded_clipped_mae,
)
import utils


def _baseline_config() -> ModelConfig:
    return ModelConfig(
        depth=3,
        base_filters=32,
        filter_growth=2.0,
        kernel_size=3,
        use_batchnorm=True,
        dropout_rate=0.25,
        l2_weight=1e-4,
        initializer="xavier",
        loss="mse",
        huber_delta=1.0,
        learning_rate=3e-4,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        batch_size=64,
        epochs=100,
        early_stop_patience=12,
        seed=559,
    )


def _run_and_record(run_name: str,
                    cfg: ModelConfig,
                    train_ds: tf.data.Dataset,
                    val_ds: tf.data.Dataset,
                    test_ds: tf.data.Dataset,
                    out_root: str,
                    aggregate_csv: str,
                    experiment: str,
                    config_label: str) -> Tuple[Dict[str, Any], tf.keras.Model, str]:
    run_dir = os.path.join(out_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(cfg)
    model = compile_model(model, cfg)
    history = train_one_run(model, train_ds, val_ds, run_dir, cfg)

    curve_path = os.path.join(run_dir, "val_mae_curves.png")
    utils.plot_history(history, curve_path, title=f"{experiment}: {config_label}")

    val_metrics = evaluate(model, val_ds)
    test_metrics = evaluate(model, test_ds)

    row: Dict[str, Any] = {
        "experiment": experiment,
        "run_name": run_name,
        "config_label": config_label,
        "loss": cfg.loss,
        "depth": cfg.depth,
        "initializer": cfg.initializer,
        "use_batchnorm": cfg.use_batchnorm,
        "dropout_rate": cfg.dropout_rate,
        "l2_weight": cfg.l2_weight,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "val_rc_mae": round(val_metrics["rc_mae"], 4),
        "val_mae": round(val_metrics["mae"], 4),
        "test_rc_mae": round(test_metrics["rc_mae"], 4),
        "test_mae": round(test_metrics["mae"], 4),
    }
    header = list(row.keys())
    utils.append_csv_row(aggregate_csv, header, row)
    utils.append_csv_row(os.path.join(out_root, f"{experiment}.csv"), header, row)
    utils.save_json(
        {"config": cfg.to_dict(), "val_metrics": val_metrics, "test_metrics": test_metrics},
        os.path.join(run_dir, "metrics.json"),
    )
    return row, model, run_dir


def _update_best(best_val: float,
                 best_record: Dict[str, Any],
                 best_path: str,
                 candidate_val: float,
                 candidate_record: Dict[str, Any],
                 candidate_model_path: str) -> Tuple[float, Dict[str, Any], str]:
    if candidate_val < best_val:
        return candidate_val, candidate_record, candidate_model_path
    return best_val, best_record, best_path


def run_experiments(args: argparse.Namespace) -> None:
    utils.set_global_seed(args.seed)
    out_root = args.output
    os.makedirs(out_root, exist_ok=True)

    train_ds, val_ds, test_ds, data_info = utils.build_datasets(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        img_size=(80, 80),
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        batch_size=args.batch_size or _baseline_config().batch_size,
    )
    utils.save_json(data_info, os.path.join(out_root, "data_info.json"))

    baseline_cfg = _baseline_config()
    if args.batch_size:
        baseline_cfg.batch_size = args.batch_size

    experiments: List[str] = args.experiments
    aggregate_csv = os.path.join(out_root, "summary.csv")

    best_val = float("inf")
    best_record: Dict[str, Any] = {}
    best_model_dir = ""

    # Architecture depth
    if "architecture" in experiments:
        for depth in [2, 3, 4]:
            cfg = _baseline_config()
            cfg.depth = depth
            cfg.batch_size = baseline_cfg.batch_size
            row, _, run_dir = _run_and_record(
                run_name=f"architecture_depth{depth}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="architecture",
                config_label=f"depth={depth}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # Loss function
    if "loss" in experiments:
        for loss in ["mse", "mae", "huber"]:
            cfg = _baseline_config()
            cfg.loss = loss
            cfg.batch_size = baseline_cfg.batch_size
            row, _, run_dir = _run_and_record(
                run_name=f"loss_{loss}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="loss",
                config_label=f"loss={loss}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # Initialisation
    if "init" in experiments:
        for initializer in ["xavier", "gaussian"]:
            cfg = _baseline_config()
            cfg.initializer = initializer
            cfg.batch_size = baseline_cfg.batch_size
            row, _, run_dir = _run_and_record(
                run_name=f"init_{initializer}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="init",
                config_label=f"init={initializer}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # Batch normalisation
    if "batchnorm" in experiments:
        for use_bn in [True, False]:
            cfg = _baseline_config()
            cfg.use_batchnorm = use_bn
            cfg.batch_size = baseline_cfg.batch_size
            tag = "on" if use_bn else "off"
            row, _, run_dir = _run_and_record(
                run_name=f"batchnorm_{tag}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="batchnorm",
                config_label=f"batchnorm={use_bn}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # L2 regularisation
    if "l2" in experiments:
        for l2_weight in [0.0, 1e-4, 1e-3]:
            cfg = _baseline_config()
            cfg.l2_weight = l2_weight
            cfg.batch_size = baseline_cfg.batch_size
            row, _, run_dir = _run_and_record(
                run_name=f"l2_{l2_weight:g}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="l2",
                config_label=f"l2={l2_weight:g}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # Dropout rate
    if "dropout" in experiments:
        for dropout in [0.0, 0.25, 0.5]:
            cfg = _baseline_config()
            cfg.dropout_rate = dropout
            cfg.batch_size = baseline_cfg.batch_size
            row, _, run_dir = _run_and_record(
                run_name=f"dropout_{dropout:g}",
                cfg=cfg,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                out_root=out_root,
                aggregate_csv=aggregate_csv,
                experiment="dropout",
                config_label=f"dropout={dropout:g}",
            )
            best_val, best_record, best_model_dir = _update_best(
                best_val, best_record, best_model_dir,
                row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
            )

    # Adam tuning
    if "adam" in experiments:
        for lr in [1e-4, 3e-4, 1e-3]:
            for batch_size in [32, 64]:
                cfg = _baseline_config()
                cfg.learning_rate = lr
                cfg.batch_size = batch_size
                row, _, run_dir = _run_and_record(
                    run_name=f"adam_lr{lr:g}_bs{batch_size}",
                    cfg=cfg,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                    out_root=out_root,
                    aggregate_csv=aggregate_csv,
                    experiment="adam",
                    config_label=f"lr={lr:g}, bs={batch_size}",
                )
                best_val, best_record, best_model_dir = _update_best(
                    best_val, best_record, best_model_dir,
                    row["val_rc_mae"], row, os.path.join(run_dir, "saved_model"),
                )

    summary = {
        "best": best_record,
        "data_info": data_info,
        "output_dir": out_root,
    }
    utils.save_json(summary, os.path.join(out_root, "best_model_summary.json"))

    if best_model_dir and os.path.exists(best_model_dir):
        src_curve = os.path.join(os.path.dirname(best_model_dir), "val_mae_curves.png")
        dst_curve = os.path.join(out_root, "validation_mae_curves.png")
        if os.path.exists(src_curve):
            shutil.copy2(src_curve, dst_curve)

        best_model = tf.keras.models.load_model(
            best_model_dir,
            custom_objects={"rounded_clipped_mae": rounded_clipped_mae},
        )
        utils.save_success_failure_examples(
            model=best_model,
            dataset=test_ds,
            success_dir=os.path.join(out_root, "success_examples"),
            failure_dir=os.path.join(out_root, "failure_examples"),
            n_each=3,
        )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS559 F2025 facial attractiveness regression experiments")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to SCUT_FBP5500_downsampled dataset root (expects train/val/test folders with labelled filenames).")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="Optional explicit path to labels CSV (used only when directory splits are absent).")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory for experiment outputs.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction.")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=559,
                        help="Random seed for reproducibility.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override baseline batch size.")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["architecture", "loss", "init", "batchnorm", "l2", "dropout", "adam"],
                        help="Experiment groups to run.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
