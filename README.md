# Clasificación Automática de Calidad de Manzanas 🍎
**Algoritmos y Programación III – ICESI 2026-1**

Sistema de visión por computadora que clasifica la calidad de manzanas (alta / media / baja) y estima su tamaño relativo, usando modelos de ML tradicional y CNN bajo la metodología CRISP-DM.

---

## Estructura del repositorio

```
Apo3_ProyectoFinal/
│
├── docs/                          # Documentación del proyecto
│   ├── arquitectura.md            # Diseño del sistema y modelos
│   └── instalacion.md             # Guía de instalación paso a paso
│
├── src/                           # Código fuente principal
│   ├── data/
│   │   └── preprocess.py          # Carga, resize, normalización, split
│   ├── models/
│   │   ├── ml_models.py           # SVM y Random Forest (GridSearchCV)
│   │   └── cnn_model.py           # CNN desde cero (Keras)
│   ├── training/
│   │   └── train.py               # Scripts de entrenamiento
│   ├── evaluation/
│   │   └── evaluate.py            # Métricas, matrices, comparativas
│   ├── utils/
│   │   ├── features.py            # Extracción de características (HSV, LBP, Hu)
│   │   └── helpers.py             # Gráficas vectoriales SVG
│   └── main.py                    # Punto de entrada principal
│
├── notebooks/                     # Jupyter Notebooks por fase CRISP-DM
│   ├── 01_comprension_datos.ipynb
│   ├── 02_preparacion_datos.ipynb
│   ├── 03_modelado_ml.ipynb
│   ├── 04_modelado_cnn.ipynb
│   └── 05_evaluacion_comparativa.ipynb
│
├── experiments/                   # Resultados de experimentos
│   ├── logs/                      # Logs de entrenamiento (CSV)
│   ├── checkpoints/               # Modelos guardados (.pkl, .h5)
│   └── results/                   # Métricas y gráficas SVG
│
├── tests/
│   └── test_models.py             # Pruebas unitarias (pytest)
│
├── data/                          # Imágenes (no se suben al repo)
│   ├── raw/                       ← imágenes originales sin tocar
│   ├── processed/                 ← imágenes preprocesadas
│   └── annotations/               ← etiquetas CSV
│
├── app/
│   └── app.py                     # Interfaz Streamlit
│
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

- **Base:** [Fruit Quality Classification – Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- **Propio:** 30–50 imágenes recolectadas en supermercado/hogar

| Clase  | Descripción |
|--------|-------------|
| `alta` | Color uniforme, sin manchas ni deformaciones |
| `media`| Manchas leves o pequeñas deformaciones |
| `baja` | Daños visibles, podredumbre o deformación severa |

---

## Instalación

Ver [`docs/instalacion.md`](docs/instalacion.md) para instrucciones detalladas.

```bash
pip install -r requirements.txt
pytest tests/                  # verificar instalación
streamlit run app/app.py       # lanzar la app
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
python src/main.py --action predict --model svm --image ruta/imagen.jpg
```

---

## Modelos

| Modelo | Tipo | Hiperparámetros ajustados |
|--------|------|--------------------------|
| SVM (kernel RBF) | ML tradicional | C, gamma – GridSearchCV 5-fold |
| Random Forest    | ML tradicional | n_estimators, max_depth, max_features |
| CNN (3 conv)     | Deep Learning  | lr, dropout, batch_size |

Ver [`docs/arquitectura.md`](docs/arquitectura.md) para detalles matemáticos.

---

## Licencia y referencias

- Código desarrollado en ICESI 2026-1.
- Dataset: Ryan Park, "Fruit Quality Classification", Kaggle, 2023.
- Todo código de terceros está referenciado explícitamente en el archivo fuente.