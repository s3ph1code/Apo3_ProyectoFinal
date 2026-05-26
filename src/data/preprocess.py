"""
preprocess.py
-------------
Scripts para cargar y preprocesar imágenes de manzanas.

Operaciones:
    - Carga de imágenes desde disco
    - Redimensionamiento y normalización
    - Data augmentation (albumentations)
    - División estratificada train / val / test
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

import albumentations as A
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (224, 224)
RANDOM_STATE: int = 42
CLASS_NAMES = ["alta", "baja", "media"]   # orden alfabético = orden de etiquetas


def load_image(path: str, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Carga una imagen en RGB, la redimensiona y normaliza a [0, 1].

    Parámetros
    ----------
    path : ruta al archivo de imagen
    size : (W, H) destino en píxeles

    Retorna
    -------
    np.ndarray (H, W, 3) float32 en [0, 1]
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def load_dataset(data_dir: str, size: Tuple[int, int] = IMG_SIZE):
    """
    Carga todas las imágenes organizadas por subcarpetas de clase.

    Estructura esperada:
        data_dir/
            alta/   img1.jpg ...
            media/  ...
            baja/   ...

    Retorna
    -------
    X          : np.ndarray (N, H, W, 3)
    y          : np.ndarray (N,) enteros
    class_names: list[str]
    """
    data_path = Path(data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    X, y = [], []
    for label, cls in enumerate(class_names):
        for img_path in (data_path / cls).glob("*.[jJpP][pPnN][gG]*"):
            try:
                X.append(load_image(str(img_path), size))
                y.append(label)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")

    print(f"[DATA] {len(X)} imágenes cargadas | Clases: {class_names}")
    return np.array(X), np.array(y), class_names


def split_dataset(X, y, test_size=0.15, val_size=0.15):
    """
    División estratificada: 70% train / 15% val / 15% test.

    Justificación matemática:
        La estratificación preserva P(y=k) en cada subconjunto,
        minimizando el sesgo en la estimación de métricas.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=rel_val, stratify=y_tr, random_state=RANDOM_STATE
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def get_augmentation(training: bool = True) -> A.Compose:
    """Pipeline de albumentations — augmentation solo en entrenamiento."""
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        ])
    return A.Compose([])
