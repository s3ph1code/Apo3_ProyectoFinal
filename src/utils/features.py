"""
features.py
-----------
Extracción de características manuales para los modelos de ML tradicional.

Características implementadas:
    - Histograma de color HSV  (96 valores)
    - Momentos de Hu           ( 7 valores)
    - Local Binary Patterns    (32 valores)
    - Estadísticas RGB         ( 6 valores)
    ─────────────────────────────────────
    Total                      141 valores
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def color_histogram_hsv(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Histograma concatenado (H, S, V) normalizado.

    El espacio HSV separa cromaticidad (H) de luminosidad (V),
    reduciendo la sensibilidad a cambios de iluminación.
    """
    img8 = (image * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
    h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h, s, v]).astype(np.float32)
    return hist / (hist.sum() + 1e-9)


def hu_moments(image: np.ndarray) -> np.ndarray:
    """
    7 momentos de Hu — invariantes a escala, rotación y traslación.

    Se aplica log10 del valor absoluto para estabilidad numérica.
    """
    img8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def lbp_histogram(image: np.ndarray, n_points: int = 24,
                  radius: int = 3, bins: int = 32) -> np.ndarray:
    """
    Histograma de Local Binary Patterns — captura textura local.

    LBP codifica cada píxel comparándolo con n_points vecinos
    en un círculo de radio `radius`.
    """
    img8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, n_points + 2))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-9)


def channel_stats(image: np.ndarray) -> np.ndarray:
    """Media y desviación estándar por canal RGB → 6 valores."""
    return np.concatenate([image.mean(axis=(0, 1)), image.std(axis=(0, 1))])


def extract_features(image: np.ndarray) -> np.ndarray:
    """Vector combinado de 141 características para ML tradicional."""
    return np.concatenate([
        color_histogram_hsv(image),
        hu_moments(image),
        lbp_histogram(image),
        channel_stats(image),
    ])
