"""
TensorFlow model definitions and training helpers for CS559 Fall 2025 facial attractiveness regression.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import tensorflow as tf


# --------------------------------------------------------------------------- #
# Custom metric
# --------------------------------------------------------------------------- #
@tf.keras.utils.register_keras_serializable(package="cs559")
def rounded_clipped_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Rounded-and-clipped MAE metric used by the assignment.

    Predictions are first clipped to [0, 10], rounded to the nearest integer, and the mean
    absolute error is taken against ground truth labels also clipped to [0, 10].
    """
    y_true = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 10.0)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 10.0)
    y_pred = tf.round(y_pred)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # Data
    input_shape: Tuple[int, int, int] = (80, 80, 3)

    # Architecture
    depth: int = 3
    base_filters: int = 32
    filter_growth: float = 2.0
    kernel_size: int = 3
    use_batchnorm: bool = True
    dropout_rate: float = 0.25
    l2_weight: float = 0.0

    # Initialization
    initializer: str = "xavier"  # {"xavier", "gaussian", "he"}

    # Loss / optimisation
    loss: str = "mse"  # {"mse", "mae", "huber"}
    huber_delta: float = 1.0
    learning_rate: float = 3e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    batch_size: int = 64
    epochs: int = 100
    early_stop_patience: int = 10

    # Misc
    seed: int = 559

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def _get_initializer(name: str, seed: Optional[int]) -> tf.keras.initializers.Initializer:
    name = (name or "").lower()
    if name in {"xavier", "glorot", "glorot_uniform"}:
        return tf.keras.initializers.GlorotUniform(seed=seed)
    if name in {"gaussian", "normal", "random_normal"}:
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)
    if name in {"he", "he_normal"}:
        return tf.keras.initializers.HeNormal(seed=seed)
    return tf.keras.initializers.GlorotUniform(seed=seed)


def _conv_block(x: tf.Tensor,
                filters: int,
                kernel_size: int,
                use_batchnorm: bool,
                dropout_rate: float,
                l2_weight: float,
                initializer: tf.keras.initializers.Initializer,
                prefix: str) -> tf.Tensor:
    regularizer = tf.keras.regularizers.l2(l2_weight) if l2_weight > 0 else None

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name=f"{prefix}_conv1",
    )(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{prefix}_relu1")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name=f"{prefix}_conv2",
    )(x)
    if use_batchnorm:
        x = tf.keras.layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{prefix}_relu2")(x)

    x = tf.keras.layers.MaxPool2D(pool_size=2, name=f"{prefix}_pool")(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{prefix}_drop")(x)
    return x


def build_model(cfg: ModelConfig) -> tf.keras.Model:
    """Construct the CNN regressor according to configuration."""
    tf.keras.utils.set_random_seed(cfg.seed)

    initializer = _get_initializer(cfg.initializer, seed=cfg.seed)

    inputs = tf.keras.Input(shape=cfg.input_shape, name="input_image")
    x = inputs
    filters = cfg.base_filters
    for block_idx in range(cfg.depth):
        x = _conv_block(
            x,
            filters=int(round(filters)),
            kernel_size=cfg.kernel_size,
            use_batchnorm=cfg.use_batchnorm,
            dropout_rate=cfg.dropout_rate,
            l2_weight=cfg.l2_weight,
            initializer=initializer,
            prefix=f"block{block_idx+1}",
        )
        filters *= cfg.filter_growth

    regularizer = tf.keras.regularizers.l2(cfg.l2_weight) if cfg.l2_weight > 0 else None
    x = tf.keras.layers.Conv2D(
        int(round(filters)),
        3,
        padding="same",
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="post_conv",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    if cfg.dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(cfg.dropout_rate, name="gap_drop")(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name="fc1",
    )(x)
    if cfg.dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(cfg.dropout_rate, name="fc1_drop")(x)

    outputs = tf.keras.layers.Dense(1, activation="linear", name="prediction", kernel_initializer=initializer)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="AttractivenessRegressor")
    return model


def _make_loss(loss_name: str, huber_delta: float) -> tf.keras.losses.Loss:
    name = (loss_name or "").lower()
    if name in {"mae", "l1"}:
        return tf.keras.losses.MeanAbsoluteError()
    if name in {"huber", "smooth_l1"}:
        return tf.keras.losses.Huber(delta=huber_delta)
    return tf.keras.losses.MeanSquaredError()


def compile_model(model: tf.keras.Model, cfg: ModelConfig) -> tf.keras.Model:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
        beta_1=cfg.adam_beta_1,
        beta_2=cfg.adam_beta_2,
    )
    loss = _make_loss(cfg.loss, cfg.huber_delta)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            rounded_clipped_mae,
        ],
    )
    return model


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def get_callbacks(output_dir: str, cfg: ModelConfig) -> List[tf.keras.callbacks.Callback]:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "best.ckpt")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_rounded_clipped_mae",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rounded_clipped_mae",
            patience=cfg.early_stop_patience,
            mode="min",
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "history.csv"), append=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def train_one_run(model: tf.keras.Model,
                  train_ds: tf.data.Dataset,
                  val_ds: tf.data.Dataset,
                  output_dir: str,
                  cfg: ModelConfig) -> tf.keras.callbacks.History:
    callbacks = get_callbacks(output_dir, cfg)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(os.path.join(output_dir, "saved_model"))
    return history


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset) -> Dict[str, float]:
    metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    return {
        "loss": float(metrics.get("loss", 0.0)),
        "mae": float(metrics.get("mae", 0.0)),
        "rc_mae": float(metrics.get("rounded_clipped_mae", 0.0)),
    }
