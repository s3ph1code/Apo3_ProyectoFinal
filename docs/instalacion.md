# Guía de instalación

## Requisitos previos

- Python 3.10 o superior
- Git

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
conda activate apple-quality
```

## Descargar el dataset

1. Ir a https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
2. Descargar y descomprimir en `data/raw/`
3. Organizar en subcarpetas por clase:
   ```
   data/raw/
       alta/   ← imágenes de alta calidad
       media/  ← imágenes de calidad media
       baja/   ← imágenes de baja calidad
   ```

## Verificar instalación

```bash
pytest tests/
```

## Ejecutar la app

```bash
streamlit run app/app.py
```
