"""
build_manifest.py
-----------------
Construye un manifest CSV unificado de todas las imágenes del proyecto.

Recorre dos fuentes:
    1. `Fruits/`             -> recolección compartida del curso APO3 ICESI.
    2. `data/raw/kaggle/`    -> descarga del dataset Kaggle (ryandpark/fruit-quality-classification).

Para cada imagen produce una fila con:
    path     : ruta relativa a la raíz del repo (POSIX, con `/`)
    fruit    : Apple | Banana | Guava | Lime | Orange | Pomegranate
    quality  : Good | Regular | Bad
    class    : Fruit_Quality (etiqueta combinada de las 18 clases)
    source   : recolectada | kaggle
    width    : ancho en píxeles
    height   : alto en píxeles

El CSV resultante es la **fuente única de verdad** consumida por los notebooks
de EDA, preprocesamiento y modelado (no se vuelven a recorrer los directorios).

Uso
---
    python src/data/build_manifest.py
    python src/data/build_manifest.py --no-kaggle      # solo recolectada
    python src/data/build_manifest.py --no-dims        # omite width/height (más rápido)
    python src/data/build_manifest.py --output other.csv

Estructura esperada de cada fuente
----------------------------------
    <fuente>/
        Good Quality_Fruits/
            Apple_Good/   *.jpg
            Banana_Good/  *.jpg
            ...
        Regular Quality_Fruits/
            Apple_Regular/  ...
        Bad Quality_Fruits/
            Apple_Bad/      ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Constantes del proyecto (alineadas con el filesystem)
# ---------------------------------------------------------------------------
FRUITS: list[str] = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate",
]

QUALITIES: list[str] = ["Good", "Regular", "Bad"]

QUALITY_CATEGORY_DIR: dict[str, str] = {
    "Good": "Good Quality_Fruits",
    "Regular": "Regular Quality_Fruits",
    "Bad": "Bad Quality_Fruits",
}

CLASS_NAMES: list[str] = sorted(
    f"{f}_{q}" for f in FRUITS for q in QUALITIES
)  # 18 etiquetas combinadas, orden lexicográfico

IMG_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CSV_FIELDS: list[str] = [
    "path",
    "fruit",
    "quality",
    "class",
    "source",
    "width",
    "height",
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """Raíz del repo: dos niveles arriba de este archivo (src/data/...)."""
    return Path(__file__).resolve().parent.parent.parent


def get_image_dimensions(path: Path) -> tuple[int, int] | None:
    """
    Lee width × height de la imagen sin cargar todos los píxeles en memoria.

    Retorna None si la imagen no se puede abrir (probable corrupción).
    """
    try:
        from PIL import Image  # importación perezosa
    except ImportError:
        print(
            "[WARN] Pillow no está instalado. "
            "Las columnas width/height quedarán vacías. "
            "Instala con: pip install Pillow",
            file=sys.stderr,
        )
        return None

    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception as exc:
        print(f"[WARN] imagen corrupta o ilegible: {path} ({exc})", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Recorrido de fuentes
# ---------------------------------------------------------------------------
def scan_source(
    base: Path, source_name: str, repo: Path, *, read_dims: bool = True
) -> Iterator[dict]:
    """
    Recorre la estructura `base/<Quality Quality_Fruits>/<Fruit_Quality>/*.jpg`
    y produce diccionarios listos para escribir en el CSV.

    Parámetros
    ----------
    base        : ruta absoluta a la raíz de la fuente.
    source_name : etiqueta para la columna `source` (e.g. 'recolectada').
    repo        : ruta absoluta del repo, usada para hacer paths relativos.
    read_dims   : si True, lee width/height con Pillow (más lento).
    """
    if not base.exists():
        print(f"[INFO] Fuente no disponible (se omite): {base}", file=sys.stderr)
        return

    for quality_label, dirname in QUALITY_CATEGORY_DIR.items():
        category_dir = base / dirname
        if not category_dir.exists():
            continue
        for fruit in FRUITS:
            fruit_dir = category_dir / f"{fruit}_{quality_label}"
            if not fruit_dir.exists():
                continue
            class_name = f"{fruit}_{quality_label}"
            for img_path in sorted(fruit_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

                w, h = ("", "")
                if read_dims:
                    dims = get_image_dimensions(img_path)
                    if dims is None:
                        continue  # imagen corrupta, no la incluimos
                    w, h = dims

                # Ruta relativa al repo, siempre con forward slash (POSIX)
                try:
                    rel = img_path.resolve().relative_to(repo)
                except ValueError:
                    # La imagen está fuera del repo: guardamos la absoluta.
                    rel = img_path.resolve()
                rel_str = str(rel).replace("\\", "/")

                yield {
                    "path": rel_str,
                    "fruit": fruit,
                    "quality": quality_label,
                    "class": class_name,
                    "source": source_name,
                    "width": w,
                    "height": h,
                }


# ---------------------------------------------------------------------------
# Resumen / auditoría
# ---------------------------------------------------------------------------
def print_summary(rows: list[dict]) -> None:
    """Imprime conteo total y desglose por (source, fruit, quality)."""
    if not rows:
        print("\n[WARN] El manifest está vacío.", file=sys.stderr)
        return

    by_class = Counter(r["class"] for r in rows)
    by_source = Counter(r["source"] for r in rows)

    print(f"\n[OK] {len(rows):,} imágenes registradas en el manifest.\n")
    print("  Por fuente:")
    for src, n in sorted(by_source.items()):
        print(f"    {src:<14} {n:>7,}")
    print("\n  Por clase (Fruit_Quality):")
    for cls in CLASS_NAMES:
        n = by_class.get(cls, 0)
        bar = "█" * (n // 100)
        print(f"    {cls:<22} {n:>6,}  {bar}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    repo = repo_root()

    parser = argparse.ArgumentParser(
        description=(
            "Construye data/annotations/manifest.csv unificando las imágenes "
            "de Fruits/ (recolección del curso) y data/raw/kaggle/."
        )
    )
    parser.add_argument(
        "--fruits-dir",
        type=Path,
        default=repo / "Fruits",
        help=f"Carpeta de la recolección compartida (por defecto: {repo}/Fruits)",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=repo / "data" / "raw" / "kaggle",
        help=(
            f"Carpeta de la descarga Kaggle "
            f"(por defecto: {repo}/data/raw/kaggle)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "annotations" / "manifest.csv",
        help=(
            f"Ruta del CSV de salida "
            f"(por defecto: {repo}/data/annotations/manifest.csv)"
        ),
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="No incluir la carpeta data/raw/kaggle/.",
    )
    parser.add_argument(
        "--no-recolectada",
        action="store_true",
        help="No incluir la carpeta Fruits/.",
    )
    parser.add_argument(
        "--no-dims",
        action="store_true",
        help="No leer width/height (mucho más rápido; deja columnas vacías).",
    )
    args = parser.parse_args()

    read_dims = not args.no_dims
    sources: list[tuple[Path, str]] = []
    if not args.no_recolectada:
        sources.append((args.fruits_dir.resolve(), "recolectada"))
    if not args.no_kaggle:
        sources.append((args.kaggle_dir.resolve(), "kaggle"))

    if not sources:
        print("[ERROR] No hay fuentes que recorrer.", file=sys.stderr)
        sys.exit(2)

    rows: list[dict] = []
    for base, name in sources:
        print(f"[INFO] Escaneando '{name}' en {base} ...", file=sys.stderr)
        rows.extend(scan_source(base, name, repo, read_dims=read_dims))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Manifest escrito en: {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
