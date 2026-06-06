# Clasificación Automática de Calidad de Frutas
**Algoritmos y Programación III – ICESI 2026-1**

Sistema de visión por computadora que clasifica la **calidad** (alta / media / baja) y la **especie** de seis frutas (manzana, banano, guayaba, lima, naranja, granada) a partir de imágenes individuales, usando modelos de ML tradicional y CNN bajo la metodología CRISP-DM.

> **Estado actual:** Fase 1 – Comprensión y preparación de los datos. El scaffold inicial estaba enfocado en manzanas; el alcance fue ampliado a multi-fruta para aprovechar el dataset compartido entre todos los grupos del curso.

---

## Alcance

- **Entrada:** Imágenes estáticas de una fruta individual sobre fondo simple.
- **Salidas:**
  1. Etiqueta combinada `Fruta_Calidad` (18 clases: 6 frutas × 3 niveles de calidad).
  2. Estimación de tamaño relativo (pequeño / mediano / grande) por área en píxeles normalizada por especie.
- **Despliegue:** Interfaz Streamlit con opción de cargar imagen o capturar desde cámara.

### Clases (18)

```
Apple_Good     Apple_Regular     Apple_Bad
Banana_Good    Banana_Regular    Banana_Bad
Guava_Good     Guava_Regular     Guava_Bad
Lime_Good      Lime_Regular      Lime_Bad
Orange_Good    Orange_Regular    Orange_Bad
Pomegranate_Good  Pomegranate_Regular  Pomegranate_Bad
```

**Mapeo a etiquetas en el informe IEEE (español):**

| Inglés (filesystem) | Español (UI / informe) |
|---|---|
| `Good` | calidad alta |
| `Regular` | calidad media |
| `Bad` | calidad baja |
| `Apple` | manzana |
| `Banana` | banano |
| `Guava` | guayaba |
| `Lime` | lima |
| `Orange` | naranja |
| `Pomegranate` | granada |

Los identificadores en el código siguen el filesystem (single source of truth); la traducción a español se aplica únicamente al renderizar en la interfaz o el informe.

---

## Datasets

| Origen | Carpeta | Tamaño | Uso |
|---|---|---|---|
| **Recolección compartida del curso** | `Fruits/` | ~9515 imágenes (3.4 GB) | Entrenamiento principal. Aporte conjunto de todos los grupos (sec. 2.2 del enunciado). **No se sube al repo.** |
| **Kaggle – Fruit Quality Classification** ([Ryan Park](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)) | `data/raw/kaggle/` | Pendiente de descarga | Base externa de referencia. Descarga reproducible con `kagglehub`. |
| **`mixed_quality/` (sub-carpeta de Kaggle)** | `data/raw/kaggle/mixed_quality/` | Pendiente | Evaluación y robustez (sec. 2.1). Requiere segmentación si hay varias frutas por foto. |

### Distribución por clase (Fruits/, verificado al 2026-06-05)

| Calidad | Apple | Banana | Guava | Lime | Orange | Pomegranate | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Good | 267 | 201 | 200 | 200 | 206 | 1585 | 2659 |
| Regular | 890 | 936 | 813 | 743 | 510 | 720 | 4612 |
| Bad | 153 | 684 | 135 | 152 | 111 | 1009 | 2244 |
| **Total** | **1310** | **1821** | **1148** | **1095** | **827** | **3314** | **9515** |

El dataset presenta **desbalanceo cruzado fruta × calidad** (Pomegranate domina, Orange_Bad escaso). La estrategia de mitigación se decide tras el EDA (notebook 01).

---

## Estructura del repositorio

```
Apo3_ProyectoFinal/
│
├── docs/                          # Documentación
│   ├── arquitectura.md            # Diseño del sistema y modelos
│   └── instalacion.md             # Guía de instalación
│
├── src/                           # Código fuente
│   ├── data/
│   │   ├── preprocess.py          # Carga, resize, normalización, split estratificado
│   │   ├── download_kaggle.py     # Descarga reproducible con kagglehub
│   │   └── build_manifest.py      # Genera manifest.csv unificado de las 18 clases
│   ├── models/
│   │   ├── ml_models.py           # SVM y Random Forest (GridSearchCV)
│   │   └── cnn_model.py           # CNN desde cero (Keras), output 18 clases
│   ├── training/train.py          # Entrenamiento
│   ├── evaluation/evaluate.py     # Métricas, matrices de confusión, comparativas
│   ├── utils/
│   │   ├── features.py            # HSV histogram, momentos de Hu, LBP
│   │   └── helpers.py             # Gráficas vectoriales SVG, mapeo EN↔ES
│   └── main.py                    # Punto de entrada
│
├── notebooks/                     # Jupyter Notebooks por fase CRISP-DM
│   ├── 01_comprension_datos.ipynb       # Fase 1 – EDA
│   ├── 02_preparacion_datos.ipynb       # Fase 2 – Preprocesamiento
│   ├── 03_modelado_ml.ipynb             # Fase 3a – SVM, Random Forest
│   ├── 04_modelado_cnn.ipynb            # Fase 3b – CNN
│   └── 05_evaluacion_comparativa.ipynb  # Fase 4 – Evaluación
│
├── experiments/                   # Resultados
│   ├── logs/                      # CSV de entrenamiento
│   ├── checkpoints/               # Modelos guardados (.pkl, .h5)
│   └── results/                   # Métricas y gráficas SVG
│
├── tests/test_models.py           # Pruebas pytest
│
├── data/                          # Imágenes (no van al repo)
│   ├── raw/
│   │   └── kaggle/                # Descarga vía kagglehub
│   ├── processed/                 # Imágenes preprocesadas
│   └── annotations/
│       └── manifest.csv           # Tabla maestra (path, fruit, quality, source, ...)
│
├── app/app.py                     # Interfaz Streamlit
│
├── Fruits/                        # ← Recolección compartida del curso (NO se sube al repo)
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Instalación

Ver [`docs/instalacion.md`](docs/instalacion.md) para instrucciones detalladas.

```bash
pip install -r requirements.txt
python src/data/download_kaggle.py    # descarga Kaggle a data/raw/kaggle/
python src/data/build_manifest.py     # genera data/annotations/manifest.csv
pytest tests/                          # verificar instalación
streamlit run app/app.py               # lanzar la app
```

---

## Flujo de trabajo

```bash
# Entrenar modelos
python src/main.py --action train --model svm
python src/main.py --action train --model rf
python src/main.py --action train --model cnn

# Evaluar y comparar
python src/main.py --action eval --all

# Predecir una imagen
python src/main.py --action predict --model cnn --image ruta/imagen.jpg
```

---

## Modelos

| Modelo | Tipo | Hiperparámetros ajustados | Output |
|---|---|---|---|
| SVM (kernel RBF) | ML tradicional | C, gamma – GridSearchCV 5-fold | 18 clases |
| Random Forest | ML tradicional | n_estimators, max_depth, max_features | 18 clases |
| CNN (3 conv) | Deep Learning | lr, dropout, batch_size | 18 clases (softmax) |

Ver [`docs/arquitectura.md`](docs/arquitectura.md) para detalles matemáticos.

---

## Metodología CRISP-DM

| Fase | Notebook / artefacto | Estado |
|---|---|---|
| 1. Comprensión del negocio | Informe IEEE §1, §2 | En redacción |
| 2. Comprensión de los datos | `notebooks/01_comprension_datos.ipynb` | En curso (Fase 1) |
| 3. Preparación de los datos | `notebooks/02_preparacion_datos.ipynb` + `src/data/preprocess.py` | Pendiente |
| 4. Modelado | `notebooks/03_modelado_ml.ipynb`, `04_modelado_cnn.ipynb` | Pendiente |
| 5. Evaluación | `notebooks/05_evaluacion_comparativa.ipynb` + `src/evaluation/evaluate.py` | Pendiente |
| 6. Despliegue | `app/app.py` (Streamlit) | Pendiente |

---

## Licencia y referencias

- Código desarrollado en ICESI 2026-1 por el grupo del curso APO3.
- Dataset base: Ryan Park, *Fruit Quality Classification*, Kaggle, 2023. [Enlace](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- Dataset complementario: imágenes recolectadas y compartidas entre todos los grupos del curso APO3 ICESI 2026-1 (sec. 2.2 del enunciado).
- Todo código de terceros está referenciado explícitamente en el archivo fuente correspondiente.
