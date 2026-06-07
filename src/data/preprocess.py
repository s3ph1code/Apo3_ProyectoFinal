"""
preprocess.py
-------------
Pipeline de preparacion de datos para el proyecto APO3 ICESI 2026-1.

Responsabilidades:
    1. Cargar `data/annotations/manifest.csv` (fuente unica de verdad).
    2. Filtrar imagenes con resolucion sospechosamente baja (< 64 px de lado).
    3. Dividir 70/15/15 train/val/test ESTRATIFICANDO por la columna `class`
       (18 combinaciones Fruit_Quality), con target `quality` (3 clases).
       Esto evita que una particion quede dominada por Pomegranate y otra
       por Lime, lo cual sesga el modelo.
    4. Persistir los splits a `data/processed/{train,val,test}_manifest.csv`.
    5. Proveer utilidades de I/O (resize 224x224, normalizacion [0,1])
       y augmentation con albumentations para uso desde notebooks/modelos.

Uso
---
    python src/data/preprocess.py
    python src/data/preprocess.py --min-side 80
    python src/data/preprocess.py --output-dir data/processed --no-save

Decisiones tecnicas
-------------------
- random_state=42 fijo para reproducibilidad (requisito academico).
- IMG_SIZE = (224, 224): tamano estandar para CNNs (compatible con
  arquitecturas preentrenadas si se decide hacer transfer learning).
- Normalizacion a [0, 1]: simple y suficiente para CNN desde cero;
  para transfer learning con ImageNet, recalibrar a media/std de ImageNet.
- Augmentation se aplica SOLO en train (decision documentada en el informe).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = ["Good", "Regular", "Bad"]
N_CLASSES: int = 3

IMG_SIZE: Tuple[int, int] = (224, 224)
MIN_SIDE: int = 64
RANDOM_STATE: int = 42
SPLIT_SIZES: Tuple[float, float, float] = (0.70, 0.15, 0.15)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST: Path = REPO_ROOT / "data" / "annotations" / "manifest.csv"
DEFAULT_OUTPUT_DIR: Path = REPO_ROOT / "data" / "processed"

REQUIRED_COLUMNS = {"path", "fruit", "quality", "class", "source", "width", "height"}


# ---------------------------------------------------------------------------
# Carga y filtrado
# ---------------------------------------------------------------------------
def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> pd.DataFrame:
    """Carga el manifest CSV y valida que tenga las columnas esperadas."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest mal formado, faltan columnas: {missing}. "
            f"Encontradas: {list(df.columns)}"
        )
    return df


def filter_small_images(df: pd.DataFrame, min_side: int = MIN_SIDE) -> pd.DataFrame:
    """
    Descarta imagenes con width o height < min_side px.

    Motivacion: el EDA detecto imagenes de hasta 13x12 px, que son thumbnails
    o capturas corruptas que afectan la calidad de las features extraidas.
    """
    n0 = len(df)
    out = df[(df["width"] >= min_side) & (df["height"] >= min_side)].copy()
    n_dropped = n0 - len(out)
    print(
        f"[preprocess] Filtrado: descartadas {n_dropped} imagenes con lado < {min_side} px "
        f"({100 * n_dropped / n0:.2f}% del total). Quedan {len(out)}."
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split estratificado
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    strat_col: str = "class",
    sizes: Tuple[float, float, float] = SPLIT_SIZES,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el manifest en train/val/test con estratificacion por `strat_col`.

    Por defecto `strat_col='class'` (las 18 combinaciones Fruit_Quality), de
    modo que cada split conserve las proporciones de cada (fruta, calidad).
    El target del modelo (`quality`, 3 clases) tambien queda estratificado.

    Implementacion: dos llamadas consecutivas a sklearn.train_test_split.

    Retorna (df_train, df_val, df_test).
    """
    from sklearn.model_selection import train_test_split

    train_size, val_size, test_size = sizes
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"sizes debe sumar 1.0, suma {sum(sizes)}")

    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[strat_col],
        random_state=random_state,
    )
    val_relative = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_relative,
        stratify=df_trainval[strat_col],
        random_state=random_state,
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Persiste los tres splits como CSVs en `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_manifest.csv",
        "val": output_dir / "val_manifest.csv",
        "test": output_dir / "test_manifest.csv",
    }
    df_train.to_csv(paths["train"], index=False)
    df_val.to_csv(paths["val"], index=False)
    df_test.to_csv(paths["test"], index=False)
    print(f"[preprocess] Splits guardados en {output_dir}")
    return paths


# ---------------------------------------------------------------------------
# I/O de imagenes y augmentation
# ---------------------------------------------------------------------------
def load_and_preprocess_image(
    path: Path | str, size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """
    Lee una imagen en disco y la deja lista para el modelo:
        - Convierte BGR (cv2) -> RGB.
        - Redimensiona a `size` (default 224x224).
        - Normaliza a float32 en [0, 1].

    Retorna ndarray shape (H, W, 3) float32 en [0, 1].
    """
    import cv2  # importacion perezosa para no penalizar tests sin cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def get_train_augmentation():
    """
    Devuelve un pipeline de augmentation para entrenamiento.

    Filosofia: el desbalanceo es leve (IR=2.06), asi que la augmentation
    es conservadora para no introducir ruido innecesario. Las transformaciones
    elegidas preservan la informacion de calidad (color, textura).

    - HorizontalFlip: ok, las frutas no tienen lateralidad relevante.
    - Rotate +-25: simula angulos de captura.
    - RandomBrightnessContrast: tolerancia a iluminacion.
    - HueSaturationValue ligero: tolerancia a balance de blancos.
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, border_mode=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
        ]
    )


def get_val_augmentation():
    """Pipeline para val/test: SIN augmentation (solo identidad)."""
    import albumentations as A

    return A.Compose([])  # no-op


# ---------------------------------------------------------------------------
# Reporte de un split
# ---------------------------------------------------------------------------
def summarize_split(name: str, df: pd.DataFrame) -> None:
    """Imprime un resumen del split: tamano, distribucion por quality y por fruit."""
    print(f"\n[{name}] {len(df):,} imagenes")
    print(f"  quality (target 3-cl): {df['quality'].value_counts(normalize=True).round(4).to_dict()}")
    print(f"  fruit (metadata):      {df['fruit'].value_counts(normalize=True).round(4).to_dict()}")
    print(f"  class (estratif. 18):  {df['class'].nunique()} clases unicas")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara los splits train/val/test del proyecto APO3 ICESI. "
            "Lee data/annotations/manifest.csv, filtra outliers de tamano, "
            "estratifica por las 18 combinaciones Fruit_Quality y persiste "
            "los splits en data/processed/."
        )
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help=f"Ruta al manifest CSV (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta destino para los splits CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-side", type=int, default=MIN_SIDE,
        help=f"Filtra imagenes con lado < N px (default: {MIN_SIDE})",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="No persistir los splits a disco (solo imprimir resumen).",
    )
    args = parser.parse_args()

    df = load_manifest(args.manifest)
    print(f"[preprocess] Manifest cargado: {len(df):,} filas")
    df = filter_small_images(df, args.min_side)
    df_train, df_val, df_test = stratified_split(df)

    summarize_split("train", df_train)
    summarize_split("val", df_val)
    summarize_split("test", df_test)

    if not args.no_save:
        save_splits(df_train, df_val, df_test, args.output_dir)


if __name__ == "__main__":
    main()
