"""
Utility functions for data loading, plotting, and result management.
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 559) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------------------------------- #
# Dataset loading helpers (CSV-based fallback)
# --------------------------------------------------------------------------- #
def _infer_label_column(header: List[str]) -> str:
    candidates = ["rating", "score", "label", "attractiveness", "attractiveness_label"]
    lower = [h.lower() for h in header]
    for cand in candidates:
        if cand in lower:
            return header[lower.index(cand)]
    return header[-1]


def load_labels_csv(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if len(header) < 2:
            raise ValueError("labels CSV must contain at least filename and label columns")
        filename_key = header[0]
        label_key = _infer_label_column(header)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            try:
                rows.append({
                    "filename": row[filename_key],
                    "label": float(row[label_key]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No valid rows parsed from {csv_path}")
    return rows


def _resolve_image_path(data_root: str, filename: str) -> str:
    direct_path = os.path.join(data_root, filename)
    if os.path.exists(direct_path):
        return direct_path

    images_path = os.path.join(data_root, "images", filename)
    if os.path.exists(images_path):
        return images_path

    target_base = os.path.splitext(filename)[0].lower()
    for root, _, files in os.walk(data_root):
        for f in files:
            if os.path.splitext(f)[0].lower() == target_base:
                return os.path.join(root, f)
    return direct_path


# --------------------------------------------------------------------------- #
# Directory-based dataset helpers (SCUT_FBP5500_downsampled in this project)
# --------------------------------------------------------------------------- #
def _parse_label_from_filename(filename: str) -> float:
    base = os.path.basename(filename)
    prefix = base.split("_", 1)[0]
    try:
        return float(prefix)
    except ValueError as exc:
        raise ValueError(f"Could not parse attractiveness label from filename '{filename}'") from exc


def _collect_images_with_labels(directory: str) -> Tuple[List[str], List[float]]:
    if not os.path.isdir(directory):
        raise ValueError(f"Expected directory '{directory}' to exist.")
    collected: List[str] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_IMAGE_EXTS:
                collected.append(os.path.join(root, fname))
    if not collected:
        raise ValueError(f"No image files discovered under '{directory}'")

    collected.sort()
    paths: List[str] = []
    labels: List[float] = []
    for path in collected:
        label = _parse_label_from_filename(os.path.basename(path))
        paths.append(path)
        labels.append(label)
    return paths, labels


def _discover_directory_splits(data_root: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    candidates = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation"],
        "test": ["test", "testing"],
    }
    for split, names in candidates.items():
        for name in names:
            path = os.path.join(data_root, name)
            if os.path.isdir(path):
                mapping[split] = path
                break
    if "train" in mapping and ("val" in mapping or "test" in mapping):
        return mapping
    return {}


# --------------------------------------------------------------------------- #
# tf.data construction
# --------------------------------------------------------------------------- #
def _load_image(path: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    return image


def _build_tf_dataset(paths: List[str],
                      labels: List[float],
                      img_size: Tuple[int, int],
                      batch_size: int,
                      shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p, img_size), tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    data_root: str,
    labels_csv: Optional[str] = None,
    img_size: Tuple[int, int] = (80, 80),
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 559,
    batch_size: int = 64,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Build train/val/test tf.data datasets.

    If the dataset directory contains train/val/test subdirectories, they are used directly
    (labels are parsed from filename prefixes). Otherwise a labels CSV is expected and the
    dataset is split randomly according to val/test fractions.
    """
    set_global_seed(seed)

    directory_splits = _discover_directory_splits(data_root)
    if directory_splits:
        train_paths, train_labels = _collect_images_with_labels(directory_splits["train"])

        if "val" in directory_splits:
            val_paths, val_labels = _collect_images_with_labels(directory_splits["val"])
        else:
            # Create validation split from training data if not provided.
            rng = np.random.default_rng(seed)
            indices = np.arange(len(train_paths))
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * val_split)))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            val_paths = [train_paths[i] for i in val_indices]
            val_labels = [train_labels[i] for i in val_indices]
            train_paths = [train_paths[i] for i in train_indices]
            train_labels = [train_labels[i] for i in train_indices]

        if "test" in directory_splits:
            test_paths, test_labels = _collect_images_with_labels(directory_splits["test"])
        else:
            # Fallback: carve out a test split from training set.
            rng = np.random.default_rng(seed + 1)
            indices = np.arange(len(train_paths))
            rng.shuffle(indices)
            n_test = max(1, int(round(len(indices) * test_split)))
            test_indices = indices[:n_test]
            remain_indices = indices[n_test:]
            test_paths = [train_paths[i] for i in test_indices]
            test_labels = [train_labels[i] for i in test_indices]
            train_paths = [train_paths[i] for i in remain_indices]
            train_labels = [train_labels[i] for i in remain_indices]

        train_ds = _build_tf_dataset(train_paths, train_labels, img_size, batch_size, shuffle=True)
        val_ds = _build_tf_dataset(val_paths, val_labels, img_size, batch_size, shuffle=False)
        test_ds = _build_tf_dataset(test_paths, test_labels, img_size, batch_size, shuffle=False)

        info = {
            "num_samples": {
                "train": len(train_paths),
                "val": len(val_paths),
                "test": len(test_paths),
            },
            "source": "directory_splits",
            "img_size": img_size,
            "label_range": [0, 10],
        }
        return train_ds, val_ds, test_ds, info

    # ------------------------------------------------------------------ #
    # CSV-driven fallback (legacy behaviour)
    # ------------------------------------------------------------------ #
    labels_csv = labels_csv or os.path.join(data_root, "labels.csv")
    rows = load_labels_csv(labels_csv)

    paths: List[str] = []
    labels: List[float] = []
    for row in rows:
        img_path = _resolve_image_path(data_root, row["filename"])
        paths.append(img_path)
        labels.append(row["label"])

    num_samples = len(paths)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(round(num_samples * test_split))
    n_val = int(round(num_samples * val_split))
    n_train = num_samples - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples for requested validation/test split.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def gather(idx: np.ndarray) -> Tuple[List[str], List[float]]:
        return [paths[i] for i in idx], [labels[i] for i in idx]

    train_paths, train_labels = gather(train_idx)
    val_paths, val_labels = gather(val_idx)
    test_paths, test_labels = gather(test_idx)

    train_ds = _build_tf_dataset(train_paths, train_labels, img_size, batch_size, shuffle=True)
    val_ds = _build_tf_dataset(val_paths, val_labels, img_size, batch_size, shuffle=False)
    test_ds = _build_tf_dataset(test_paths, test_labels, img_size, batch_size, shuffle=False)

    info = {
        "num_samples": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        },
        "source": "csv_split",
        "splits": {
            "val": val_split,
            "test": test_split,
        },
        "img_size": img_size,
        "labels_csv": labels_csv,
        "label_range": [0, 10],
    }
    return train_ds, val_ds, test_ds, info


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_csv_row(path: str, header: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_history(history: tf.keras.callbacks.History, out_path: str, title: str) -> None:
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    plt.figure(figsize=(6.0, 4.0), dpi=150)
    if hist.get("mae"):
        plt.plot(epochs, hist["mae"], label="Train MAE")
    if hist.get("rounded_clipped_mae"):
        plt.plot(epochs, hist["rounded_clipped_mae"], label="Train RC-MAE")
    if hist.get("val_mae"):
        plt.plot(epochs, hist["val_mae"], label="Val MAE")
    if hist.get("val_rounded_clipped_mae"):
        plt.plot(epochs, hist["val_rounded_clipped_mae"], label="Val RC-MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------------------------------------------------------- #
# Qualitative examples
# --------------------------------------------------------------------------- #
def _overlay_caption(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    text_w, text_h = draw.textlength(text, font=font), 16
    pad = 4
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0, 128))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return image


def save_success_failure_examples(model: tf.keras.Model,
                                  dataset: tf.data.Dataset,
                                  success_dir: str,
                                  failure_dir: str,
                                  n_each: int = 3) -> None:
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    all_imgs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for batch_imgs, batch_labels in dataset:
        preds = model.predict(batch_imgs, verbose=0)
        all_imgs.append(batch_imgs.numpy())
        all_labels.append(batch_labels.numpy())
        all_preds.append(preds)

    images_np = np.concatenate(all_imgs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).reshape(-1)
    preds_np = np.concatenate(all_preds, axis=0).reshape(-1)

    preds_rc = np.clip(np.rint(np.clip(preds_np, 0.0, 10.0)), 0.0, 10.0)
    abs_err = np.abs(labels_np - preds_rc)

    success_idx = np.argsort(abs_err)[:n_each]
    failure_idx = np.argsort(-abs_err)[:n_each]

    def _save_examples(idxs: np.ndarray, directory: str) -> None:
        for i, idx in enumerate(idxs):
            img = (images_np[idx] * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img)
            caption = f"GT {labels_np[idx]:.1f} | Pred {preds_np[idx]:.2f} (rc {preds_rc[idx]:.0f})"
            pil_img = _overlay_caption(pil_img, caption)
            pil_img.save(os.path.join(directory, f"example_{i+1}.png"))

    _save_examples(success_idx, success_dir)
    _save_examples(failure_idx, failure_dir)

