# Guía de instalación

## Requisitos previos

- Python 3.10 o superior
- Git
- Cuenta de Kaggle con token API (`~/.kaggle/kaggle.json`) para descargar el dataset de referencia.

## Pasos

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPO>
cd Apo3_ProyectoFinal

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux / Mac
.venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Alternativa con conda)
conda env create -f environment.yml
conda activate fruit-quality
```

## Configurar el token de Kaggle (una sola vez)

1. Ir a https://www.kaggle.com/settings/account → sección **API** → *Create New Token*.
2. Guardar el archivo `kaggle.json` descargado en:
   - Linux / Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<usuario>\.kaggle\kaggle.json`
3. (Linux / Mac) Restringir permisos: `chmod 600 ~/.kaggle/kaggle.json`.

## Descargar el dataset Kaggle

```bash
python src/data/download_kaggle.py
```

El script usa `kagglehub` para descargar `ryandpark/fruit-quality-classification` y copia las imágenes a `data/raw/kaggle/`, manteniendo la estructura original:

```
data/raw/kaggle/
    Good Quality_Fruits/
        Apple_Good/   Banana_Good/   Guava_Good/   Lime_Good/   Orange_Good/   Pomegranate_Good/
    Regular Quality_Fruits/
        Apple_Regular/   ...
    Bad Quality_Fruits/
        Apple_Bad/   ...
    mixed_quality/   ← imágenes con varias frutas (uso: evaluación / robustez)
```

## Dataset compartido del curso

La carpeta `Fruits/` (en la raíz del repo local, **no versionada en Git** por tamaño) contiene la recolección compartida entre todos los grupos del curso (~9515 imágenes, 3.4 GB). Cada estudiante debe pedir acceso a esta carpeta a través del mecanismo definido por el curso.

## Construir el manifest unificado

```bash
python src/data/build_manifest.py
```

Recorre `Fruits/` y `data/raw/kaggle/`, anota cada imagen con su clase, fruta, calidad y fuente, y genera `data/annotations/manifest.csv`. Este CSV es la fuente única de verdad para los notebooks de EDA, preprocesamiento y modelado.

## Verificar instalación

```bash
pytest tests/
```

## Ejecutar la app

```bash
streamlit run app/app.py
```
