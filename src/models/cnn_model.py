"""
cnn_model.py
------------
CNN simple desde cero (3 bloques convolucionales) para clasificacion de
3 clases (Good / Regular / Bad). Diseno: la red no recibe la fruta como
input; debe aprender pistas universales de calidad.

Arquitectura
------------
Input  (224, 224, 3)
Block1: Conv2D(32, 3x3) + BN + ReLU + MaxPool(2x2)   -> 112x112x32
Block2: Conv2D(64, 3x3) + BN + ReLU + MaxPool(2x2)   ->  56x56x64
Block3: Conv2D(128, 3x3)+ BN + ReLU + MaxPool(2x2)   ->  28x28x128
Head:   Flatten -> Dense(256) + ReLU + Dropout(0.5)
        Dense(3, softmax)

Optimizador: Adam (lr ajustable). Perdida: categorical_crossentropy.
"""
from __future__ import annotations

from typing import Optional

CLASS_NAMES = ["Good", "Regular", "Bad"]
N_CLASSES = 3
IMG_SIZE = (224, 224)


def build_cnn(
    input_shape: tuple[int, int, int] = (*IMG_SIZE, 3),
    n_classes: int = N_CLASSES,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
):
    """
    Construye el modelo Keras. Llamar despues de importar tensorflow para
    evitar el costo de import cuando solo se necesitan utilidades.
    """
    from tensorflow.keras import layers, models, optimizers

    model = models.Sequential([
        layers.Input(shape=input_shape, name="input"),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.ReLU(name="relu1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.ReLU(name="relu2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.ReLU(name="relu3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # Head
        layers.Flatten(name="flatten"),
        layers.Dense(dense_units, activation="relu", name="dense1"),
        layers.Dropout(dropout_rate, name="dropout"),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ], name="fruit_quality_cnn")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_training_callbacks(
    checkpoint_path: str,
    patience: int = 7,
    monitor: str = "val_loss",
):
    """Callbacks estandar: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau."""
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    return [
        EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]


def compute_class_weights(y_train) -> dict:
    """
    Class weight inverso a la frecuencia: w_k = N / (K * n_k).
    Util para mitigar el desbalanceo leve (IR=2.06).
    """
    import numpy as np
    from collections import Counter

    counts = Counter(y_train)
    N = sum(counts.values())
    K = len(counts)
    # Mapeo de clase -> indice numerico (0=Good, 1=Regular, 2=Bad por orden alfab?)
    # Devolvemos pesos por indice de clase, asumiendo que keras usa indices.
    # El caller debe pasarnos y_train ya en formato indice.
    weights = {k: N / (K * v) for k, v in counts.items()}
    return weights
