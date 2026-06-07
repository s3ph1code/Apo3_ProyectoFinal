"""
size_estimator.py
-----------------
Estimacion de tamano relativo (pequeno/mediano/grande) por especie.

Razonamiento: "grande" no es comparable entre granada y lima. Calibramos
umbrales por especie con percentiles 33 y 66 del area frutal en el train.

Pipeline de calculo de area:
    1. Convertir a HSV.
    2. Umbralizar canal V (luminosidad) para separar fruta de fondo claro.
    3. Cierre morfologico para tapar huecos.
    4. Contar pixeles "fruta" (region grande).

Funciones publicas
------------------
- compute_fruit_area(img_rgb): area en proporcion del total de pixeles.
- fit_size_thresholds(df): dict {fruit: (q33, q66)} sobre train set.
- estimate_size(img_rgb, fruit, thresholds): "pequeno"|"mediano"|"grande".
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
SIZE_LABELS = ["pequeno", "mediano", "grande"]


def compute_fruit_area(img_rgb: np.ndarray, v_threshold: int = 200) -> float:
    """
    Proporcion de pixeles que son fruta (no fondo claro).

    Parametros
    ----------
    img_rgb : (H, W, 3) en cualquier rango [0, 1] o [0, 255].
    v_threshold : umbral V (0-255). Pixeles con V > umbral se consideran fondo.

    Retorna
    -------
    float en [0, 1]: fraccion de pixeles que son fruta.
    """
    import cv2

    if img_rgb.dtype != np.uint8:
        if img_rgb.max() <= 1.0:
            img_uint = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint = img_rgb.clip(0, 255).astype(np.uint8)
    else:
        img_uint = img_rgb

    bgr = cv2.cvtColor(img_uint, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Mascara: fruta = canal V < umbral (fondo claro tipicamente blanco)
    mask = (hsv[..., 2] < v_threshold).astype(np.uint8)
    # Cierre morfologico para tapar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float(mask.sum() / mask.size)


def fit_size_thresholds(
    image_paths: Iterable[str | Path],
    fruits: Iterable[str],
    image_loader=None,
    v_threshold: int = 200,
) -> dict[str, tuple[float, float]]:
    """
    Calcula percentiles 33 y 66 del area frutal por especie sobre un conjunto
    de imagenes (idealmente train).

    Parametros
    ----------
    image_paths : iterable de rutas a imagenes.
    fruits      : iterable paralelo con la especie de cada imagen.
    image_loader: funcion path -> ndarray RGB. Si None, usa
                  src.data.preprocess.load_and_preprocess_image.
    v_threshold : umbral pasado a compute_fruit_area.

    Retorna
    -------
    dict {fruit: (q33, q66)} con los percentiles de area por especie.
    """
    if image_loader is None:
        from src.data.preprocess import load_and_preprocess_image as image_loader

    areas_per_fruit: dict[str, list[float]] = {f: [] for f in FRUITS}
    for path, fruit in zip(image_paths, fruits):
        try:
            img = image_loader(path)
        except Exception:
            continue
        a = compute_fruit_area(img, v_threshold=v_threshold)
        if fruit in areas_per_fruit:
            areas_per_fruit[fruit].append(a)

    thresholds = {}
    for fruit, areas in areas_per_fruit.items():
        if len(areas) < 5:
            # Fallback: usa percentiles globales si no hay suficientes muestras
            thresholds[fruit] = (0.20, 0.45)
            continue
        arr = np.array(areas)
        thresholds[fruit] = (float(np.quantile(arr, 0.33)),
                             float(np.quantile(arr, 0.66)))
    return thresholds


def estimate_size(
    img_rgb: np.ndarray,
    fruit: str,
    thresholds: dict[str, tuple[float, float]],
    v_threshold: int = 200,
) -> str:
    """Etiqueta de tamano para una imagen dada su especie."""
    if fruit not in thresholds:
        raise ValueError(f"Fruta '{fruit}' sin umbrales calibrados.")
    area = compute_fruit_area(img_rgb, v_threshold=v_threshold)
    q33, q66 = thresholds[fruit]
    if area < q33:
        return "pequeno"
    if area < q66:
        return "mediano"
    return "grande"
