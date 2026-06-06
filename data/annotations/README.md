# `data/annotations/`

Etiquetas y manifests del proyecto. **Esta carpeta sí se versiona** (a diferencia de `data/raw/` y `data/processed/` que contienen imágenes binarias).

## `manifest.csv`

Tabla maestra que indexa todas las imágenes del proyecto. Es la **fuente única de verdad** para los notebooks de EDA, preprocesamiento, entrenamiento y evaluación: ningún script debe recorrer los directorios `Fruits/` o `data/raw/kaggle/` directamente, solo leer este CSV.

### Esquema

| Columna | Tipo | Descripción |
|---|---|---|
| `path` | str (POSIX) | Ruta relativa a la raíz del repo (e.g. `Fruits/Good Quality_Fruits/Apple_Good/img_001.jpg`). |
| `fruit` | str | Una de: `Apple`, `Banana`, `Guava`, `Lime`, `Orange`, `Pomegranate`. **No es target**, es metadata para análisis de sesgo. |
| `quality` | str | **TARGET del modelo**. Una de: `Good`, `Regular`, `Bad`. |
| `class` | str | Etiqueta combinada `Fruit_Quality` (18 combinaciones). **Solo se usa para estratificar splits**, no como target. |
| `source` | str | `recolectada` (carpeta `Fruits/`) o `kaggle` (`data/raw/kaggle/`). |
| `width` | int | Ancho en píxeles. |
| `height` | int | Alto en píxeles. |

### `quality` vs `class`: ¿cuál usar como target?

- **Modelo:** entrena y predice `quality` (3 clases: Good / Regular / Bad).
- **Split estratificado:** usa `class` (18 combinaciones) como argumento `stratify=` en `train_test_split`. Esto evita que una partición quede dominada por Pomegranate y otra por Lime.

Ejemplo correcto:

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/annotations/manifest.csv")

X = df["path"]            # rutas a las imágenes
y = df["quality"]         # target: 3 clases
strat = df["class"]       # estratificación: 18 combinaciones

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.15, stratify=strat, random_state=42
)
```

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
