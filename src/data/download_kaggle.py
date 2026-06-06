"""
download_kaggle.py
------------------
Descarga reproducible del dataset Fruit Quality Classification (Ryan Park, Kaggle).

Uso
---
    python src/data/download_kaggle.py
    python src/data/download_kaggle.py --target data/raw/kaggle
    python src/data/download_kaggle.py --force

Requisitos
----------
- kagglehub >= 0.2.0 (ver requirements.txt)
- Token API de Kaggle en ~/.kaggle/kaggle.json
  (https://www.kaggle.com/settings/account -> API -> Create New Token)

Comportamiento
--------------
1. Descarga el dataset 'ryandpark/fruit-quality-classification' a la caché local
   de kagglehub (~/.cache/kagglehub/...).
2. Copia (o crea symlinks) el contenido a `data/raw/kaggle/` dentro del repo,
   preservando la estructura original:

       data/raw/kaggle/
           Good Quality_Fruits/    Apple_Good/  Banana_Good/  ...
           Regular Quality_Fruits/ Apple_Regular/  ...
           Bad Quality_Fruits/     Apple_Bad/  ...
           mixed_quality/          (evaluación / robustez)

3. Imprime un resumen de conteos por (especie × calidad) para auditoría.

Referencia
----------
Ryan Park, "Fruit Quality Classification", Kaggle, 2023.
URL: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
KAGGLE_DATASET_ID: str = "ryandpark/fruit-quality-classification"
KAGGLE_DATASET_URL: str = (
    "https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification"
)

# Carpeta destino dentro del repo (relativa a la raíz del proyecto)
DEFAULT_TARGET: Path = Path("data/raw/kaggle")

# Extensiones de imagen consideradas válidas
IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Lógica
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """Retorna la raíz del repo (parent de src/)."""
    return Path(__file__).resolve().parent.parent.parent


def check_kaggle_credentials() -> None:
    """
    Verifica que ~/.kaggle/kaggle.json exista y dé instrucciones si no.

    kagglehub usa el mismo token que el CLI oficial de Kaggle.
    """
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        print(
            "\n[ERROR] No se encontró el token de Kaggle en:\n"
            f"  {token}\n\n"
            "Pasos para configurarlo (una sola vez):\n"
            "  1. Entra a https://www.kaggle.com/settings/account\n"
            "  2. Sección 'API' -> 'Create New Token' -> descarga kaggle.json\n"
            f"  3. Mueve el archivo a: {token}\n"
            "  4. (Linux/Mac) chmod 600 ~/.kaggle/kaggle.json\n",
            file=sys.stderr,
        )
        sys.exit(2)


def download_dataset() -> Path:
    """
    Descarga el dataset vía kagglehub y retorna la ruta de caché.

    kagglehub maneja el cacheo: si ya está descargado, retorna la ruta existente
    sin volver a bajarlo.
    """
    try:
        import kagglehub
    except ImportError:
        print(
            "[ERROR] Falta el paquete 'kagglehub'.\n"
            "Instálalo con:  pip install kagglehub>=0.2.0",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"[INFO] Descargando dataset: {KAGGLE_DATASET_ID}")
    print(f"[INFO] Fuente: {KAGGLE_DATASET_URL}")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    print(f"[OK] Dataset disponible en caché: {cache_path}")
    return cache_path


def copy_to_repo(source: Path, target: Path, force: bool = False) -> None:
    """
    Copia el contenido descargado a `target`, dentro del repo.

    Parámetros
    ----------
    source : ruta de la caché de kagglehub
    target : ruta destino (típicamente data/raw/kaggle/)
    force  : si True, sobrescribe el contenido existente en `target`
    """
    if target.exists() and any(target.iterdir()):
        if not force:
            print(
                f"[INFO] El destino {target} ya contiene datos. "
                "Usa --force para sobrescribir. Saltando copia."
            )
            return
        print(f"[WARN] --force activo: limpiando {target}")
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Copiando {source} -> {target} ...")
    # copytree con dirs_exist_ok=True para que funcione si target existe vacío
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"[OK] Copia completa.")


def summarize_counts(root: Path) -> None:
    """
    Imprime un conteo de imágenes por (categoría_calidad, fruta) recorriendo
    la estructura esperada: <root>/<X Quality_Fruits>/<Fruta_Calidad>/*.jpg
    """
    print(f"\n[INFO] Resumen de conteos en {root}:\n")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mixed_count = 0

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        # mixed_quality es una excepción: imágenes directas, sin subdivisión
        if "mixed" in name.lower():
            mixed_count = sum(
                1
                for p in sub.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            continue
        # X Quality_Fruits/ -> contiene subcarpetas Apple_X, Banana_X, ...
        for fruit_dir in sorted(sub.iterdir()):
            if not fruit_dir.is_dir():
                continue
            n = sum(
                1
                for p in fruit_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            counts[name][fruit_dir.name] = n

    total = 0
    for quality_cat, fruits in counts.items():
        cat_total = sum(fruits.values())
        total += cat_total
        print(f"  {quality_cat}  (total: {cat_total})")
        for fruit, n in sorted(fruits.items()):
            print(f"    - {fruit:<25} {n:>6}")
    if mixed_count:
        print(f"\n  mixed_quality (total: {mixed_count})")
        total += mixed_count
    print(f"\n  TOTAL: {total} imágenes\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga reproducible del dataset Kaggle "
            "Fruit Quality Classification para el proyecto APO3 ICESI."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Carpeta destino dentro del repo. "
            f"Por defecto: <repo>/{DEFAULT_TARGET}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribir el destino si ya contiene datos.",
    )
    args = parser.parse_args()

    target = args.target if args.target else repo_root() / DEFAULT_TARGET
    target = target.resolve()

    check_kaggle_credentials()
    cache_path = download_dataset()
    copy_to_repo(cache_path, target, force=args.force)
    summarize_counts(target)


if __name__ == "__main__":
    main()
