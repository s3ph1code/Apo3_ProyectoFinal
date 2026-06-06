# Clasificación Automática de Calidad de Frutas
**Algoritmos y Programación III – ICESI 2026-1**

Sistema de visión por computadora que clasifica la **calidad** de una fruta (alta / media / baja, equivalente a Good / Regular / Bad) a partir de una imagen individual, agnóstico a la especie. El dataset cubre seis frutas (manzana, banano, guayaba, lima, naranja, granada), pero el modelo aprende **pistas universales de calidad** transferibles entre especies. Se siguen modelos de ML tradicional y CNN bajo la metodología CRISP-DM.

> **Estado actual:** Fase 1 (Comprensión de los datos) terminada. Próximo paso: Fase 2 (Preparación de los datos).

---

## Alcance

- **Entrada:** Imagen estática de una fruta individual sobre fondo simple.
- **Salida del modelo:** Una de las **3 clases universales de calidad**:
  - `Good` → calidad alta (color uniforme, sin manchas ni deformaciones).
  - `Regular` → calidad media (manchas leves o pequeñas deformaciones).
  - `Bad` → calidad baja (daños visibles, podredumbre o deformación severa).
- **Salida secundaria:** Estimación de tamaño relativo (pequeño / mediano / grande) por área en píxeles, normalizada por especie.
- **Despliegue:** Interfaz Streamlit con opción de cargar imagen o capturar desde cámara.

### ¿Por qué 3 clases universales y no 18?

El profesor confirmó que el target son tres niveles de calidad **agnósticos a la especie**. La especie sigue siendo metadata importante para:
- **Estratificar los splits** train/val/test por las 18 combinaciones `Fruta_Calidad` (evita sesgos por fruta dominante).
- **Análisis de sesgo en el informe IEEE:** ¿el modelo aprende calidad real o atajos por especie? Se reporta también una matriz de confusión 6×6 por fruta para responder esto.

---

## Dataset

| Origen | Carpeta | Tamaño | Uso |
|---|---|---|---|
| **Recolección compartida del curso** | `Fruits/` | ~9515 imágenes (3.4 GB) | Entrenamiento principal. Aporte conjunto de todos los grupos (sec. 2.2 del enunciado). **No se sube al repo.** |
| **Kaggle – Fruit Quality Classification** ([Ryan Park](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)) | `data/raw/kaggle/` | Pendiente | Base externa de referencia (sec. 2.1). Descarga reproducible con `kagglehub`. Se incorporará si sobra tiempo. |
| **`mixed_quality/`** (sub-carpeta de Kaggle) | `data/raw/kaggle/mixed_quality/` | Pendiente | Evaluación y robustez. Requiere segmentación si hay varias frutas por foto. |

### Distribución de las 3 clases (sobre Fruits/)

| Clase | n | % |
|---|---:|---:|
| Good | 2 659 | 27.9% |
| Regular | 4 612 | 48.5% |
| Bad | 2 244 | 23.6% |
| **Total** | **9 515** | 100% |

Imbalance Ratio = 2.06, entropía normalizada = 0.954 (cerca de uniforme). El desbalanceo es **leve** y manejable sin técnicas agresivas.

### Distribución cruzada fruta × calidad (para estratificación y análisis de sesgo)

| Calidad | Apple | Banana | Guava | Lime | Orange | Pomegranate |
|---|---:|---:|---:|---:|---:|---:|
| Good | 267 | 201 | 200 | 200 | 206 | 1 585 |
| Regular | 890 | 936 | 813 | 743 | 510 | 720 |
| Bad | 153 | 684 | 135 | 152 | 111 | 1 009 |

**Pomegranate domina en Good y Bad**, lo que motiva el split estratificado por las 18 combinaciones.

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
│   │   └── build_manifest.py      # Genera manifest.csv unificado
│   ├── models/
│   │   ├── ml_models.py           # SVM y Random Forest (GridSearchCV)
│   │   └── cnn_model.py           # CNN desde cero (Keras), output 3 clases
│   ├── training/train.py
│   ├── evaluation/evaluate.py     # Métricas + matriz 3×3 (target) + 6×6 (sesgo por fruta)
│   ├── utils/
│   │   ├── features.py            # HSV histogram, momentos de Hu, LBP (141-D)
│   │   └── helpers.py             # Gráficas vectoriales SVG, mapeo EN↔ES
│   └── main.py                    # Punto de entrada
│
├── notebooks/                     # Jupyter Notebooks por fase CRISP-DM
│   ├── 01_comprension_datos.ipynb       ✓ Fase 1 – EDA
│   ├── 02_preparacion_datos.ipynb       ⌛ Fase 2 – Preprocesamiento
│   ├── 03_modelado_ml.ipynb             ⌛ Fase 3a – SVM, Random Forest
│   ├── 04_modelado_cnn.ipynb            ⌛ Fase 3b – CNN
│   └── 05_evaluacion_comparativa.ipynb  ⌛ Fase 4 – Evaluación comparativa
│
├── experiments/
│   ├── logs/                      # CSV de entrenamiento
│   ├── checkpoints/               # Modelos guardados (.pkl, .h5)
│   └── results/                   # Métricas exportadas
│
├── tests/test_models.py
│
├── data/                          # Imágenes (no van al repo)
│   ├── raw/kaggle/                # Descarga vía kagglehub
│   ├── processed/                 # Imágenes preprocesadas (224×224)
│   └── annotations/
│       ├── manifest.csv           # Tabla maestra (versionada)
│       └── README.md
│
├── outputs/
│   ├── figures/eda/               # 8+ SVGs vectoriales para el informe
│   └── eda_report.md              # Síntesis Fase 1
│
├── app/app.py                     # Interfaz Streamlit
│
├── Fruits/                        # ← Recolección compartida (NO se sube al repo)
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
python -m venv .venv
source .venv/Scripts/activate         # Windows Git Bash
# o:  .\.venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt
python src/data/build_manifest.py     # genera data/annotations/manifest.csv
jupyter lab notebooks/01_comprension_datos.ipynb
```

---

## Flujo de trabajo

```bash
# Entrenar modelos (Fase 3 — pendiente de implementación)
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

| Modelo | Tipo | Output | Hiperparámetros ajustados |
|---|---|---|---|
| SVM (kernel RBF) | ML tradicional | 3 clases (one-vs-rest) | C, gamma — GridSearchCV 5-fold |
| Random Forest | ML tradicional | 3 clases | n_estimators, max_depth, max_features |
| CNN (3 conv) | Deep Learning | 3 clases (softmax) | lr, dropout, batch_size |

Ver [`docs/arquitectura.md`](docs/arquitectura.md) para los detalles matemáticos.

---

## Metodología CRISP-DM

| Fase | Notebook / artefacto | Estado |
|---|---|---|
| 1. Comprensión del negocio | Informe IEEE §1, §2 | En redacción |
| 2. Comprensión de los datos | `notebooks/01_comprension_datos.ipynb` + `outputs/eda_report.md` | **Terminada** |
| 3. Preparación de los datos | `notebooks/02_preparacion_datos.ipynb` + `src/data/preprocess.py` | En curso |
| 4. Modelado | `notebooks/03_modelado_ml.ipynb`, `04_modelado_cnn.ipynb` | Pendiente |
| 5. Evaluación | `notebooks/05_evaluacion_comparativa.ipynb` + `src/evaluation/evaluate.py` | Pendiente |
| 6. Despliegue | `app/app.py` (Streamlit) | Pendiente |

---

## Licencia y referencias

- Código desarrollado en ICESI 2026-1 por el grupo del curso APO3.
- Dataset base: Ryan Park, *Fruit Quality Classification*, Kaggle, 2023. [Enlace](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- Dataset complementario: imágenes recolectadas y compartidas entre todos los grupos del curso APO3 ICESI 2026-1 (sec. 2.2 del enunciado).
- Todo código de terceros está referenciado explícitamente en el archivo fuente correspondiente.
