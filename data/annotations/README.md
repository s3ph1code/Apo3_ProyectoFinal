# `data/annotations/`

Etiquetas y manifests del proyecto. **Esta carpeta sí se versiona** (a diferencia de `data/raw/` y `data/processed/` que contienen imágenes binarias).

## `manifest.csv`

Tabla maestra que indexa todas las imágenes disponibles para el proyecto. Es la **fuente única de verdad** para los notebooks de EDA, preprocesamiento y modelado: ningún script debe recorrer los directorios `Fruits/` o `data/raw/kaggle/` directamente, solo leer este CSV.

### Esquema

| Columna | Tipo | Descripción |
|---|---|---|
| `path` | str (POSIX) | Ruta relativa a la raíz del repo (e.g. `Fruits/Good Quality_Fruits/Apple_Good/img_001.jpg`). |
| `fruit` | str | Una de: `Apple`, `Banana`, `Guava`, `Lime`, `Orange`, `Pomegranate`. |
| `quality` | str | Una de: `Good`, `Regular`, `Bad`. |
| `class` | str | Etiqueta combinada `Fruit_Quality` (18 clases). |
| `source` | str | `recolectada` (carpeta `Fruits/`) o `kaggle` (`data/raw/kaggle/`). |
| `width` | int | Ancho en píxeles. |
| `height` | int | Alto en píxeles. |

### Regenerar

```bash
python src/data/build_manifest.py
```

Banderas útiles:

- `--no-kaggle` / `--no-recolectada` para incluir solo una fuente.
- `--no-dims` para omitir width/height (mucho más rápido).
- `--output other.csv` para escribir a otra ruta.

### Versionado

El CSV se versiona como **snapshot** del estado del dataset al momento del último commit que lo modifique. Si el dataset cambia (más imágenes, eliminaciones), regenera el manifest y vuelve a commitearlo.
