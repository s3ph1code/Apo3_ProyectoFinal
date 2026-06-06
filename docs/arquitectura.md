# Arquitectura del sistema

## Alcance

Clasificación de **calidad (Good / Regular / Bad, 3 clases universales)** sobre seis frutas (manzana, banano, guayaba, lima, naranja, granada). El modelo es **agnóstico a la especie**: una sola etiqueta de tres niveles describe el estado de calidad sin distinguir qué fruta es. La especie se mantiene como metadata para análisis de sesgo y para estratificación de los splits.

## Pipeline general (CRISP-DM)

```
[Fruits/   recolección del curso, 9515 imgs]
[data/raw/kaggle/   descarga opcional con kagglehub]
                  │
                  ▼
   [src/data/build_manifest.py]
   → data/annotations/manifest.csv
     columnas: path, fruit, quality, class, source, width, height
     ─ quality = TARGET (3 clases)
     ─ class   = Fruit_Quality (18) usado solo para estratificación
                  │
                  ▼
   [src/data/preprocess.py]
     - Resize 224×224
     - Normalización [0, 1]
     - Split estratificado por `class` (18) → 70/15/15
     - Data augmentation (solo en train)
                  │
        ┌─────────┴──────────────────┐
        ▼                            ▼
[src/utils/features.py]    [Píxeles directos (CNN)]
  - HSV histogram (96)
  - Momentos de Hu (7)
  - LBP histogram (32)
  - Channel stats (6)
  → vector 141-D
        │                            │
        ▼                            ▼
[src/models/ml_models.py]   [src/models/cnn_model.py]
  - SVM kernel RBF            - 3 bloques Conv2D + BN + Pool
  - Random Forest             - Dense(256) + Dropout(0.5)
  - GridSearchCV 5-fold       - Softmax(3 clases)
        │                            │
        └────────────┬───────────────┘
                     ▼
         [src/evaluation/evaluate.py]
           - Classification report (3 clases)
           - Matriz de confusión 3×3 (target)
           - Matriz 6×6 marginal por fruta (análisis de sesgo)
           - Análisis condicional: ¿el modelo es uniformemente bueno entre frutas?
           - Curvas de aprendizaje (SVG)
                     │
                     ▼
              [app/app.py]
           Interfaz Streamlit
```

## Modelos implementados

### SVM con kernel RBF (multiclase one-vs-rest)

Optimización del margen suave en el espacio de características:

```
min_{w,b}  ½‖w‖² + C Σᵢ ξᵢ
sujeto a:  yᵢ(w·φ(xᵢ) + b) ≥ 1 − ξᵢ,    ξᵢ ≥ 0
```

Kernel gaussiano: `k(x, x') = exp(−γ ‖x − x'‖²)`.

Para 3 clases se aplica estrategia *one-vs-rest*: tres clasificadores binarios; la clase predicha es la del mayor score:

```
ŷ = argmax_{k ∈ {Good, Regular, Bad}}  fₖ(x)
```

### Random Forest

Ensemble de T árboles de decisión entrenados con bagging y selección aleatoria de características por nodo:

```
ŷ = argmax_{k ∈ {Good, Regular, Bad}}  Σ_{t=1}^{T} 𝟙[hₜ(x) = k]
```

Impureza de Gini sobre 3 clases:

```
Gini(S) = 1 − Σ_{k=1}^{3} pₖ²
```

### CNN (3 bloques convolucionales)

```
Input (224×224×3)
→ Conv2D(32, 3×3) + BN + ReLU + MaxPool(2×2)   →  112×112×32
→ Conv2D(64, 3×3) + BN + ReLU + MaxPool(2×2)   →   56×56×64
→ Conv2D(128, 3×3) + BN + ReLU + MaxPool(2×2)  →   28×28×128
→ Flatten → Dense(256) + ReLU → Dropout(0.5)
→ Dense(3) + Softmax
```

Pérdida: entropía cruzada categórica

```
ℒ = − (1/N) Σᵢ Σ_{k=1}^{3} yᵢₖ log(ŷᵢₖ)
```

El desbalanceo es leve (Imbalance Ratio = 2.06), pero por consistencia aplicamos `class_weight` inverso a la frecuencia:

```
wₖ = N / (3 · nₖ)      donde nₖ es el número de ejemplos de la clase k
```

## Etiquetas y mapeo

Identificadores en código (filesystem-aligned):

```python
CLASS_NAMES = ["Good", "Regular", "Bad"]   # target del modelo
N_CLASSES   = 3
```

Traducción a español (solo en UI / informe IEEE):

| Inglés | Español |
|---|---|
| Good | calidad alta |
| Regular | calidad media |
| Bad | calidad baja |

Para reportar resultados desagregados por especie:

| Inglés | Español |
|---|---|
| Apple | manzana |
| Banana | banano |
| Guava | guayaba |
| Lime | lima |
| Orange | naranja |
| Pomegranate | granada |

## Estratificación del split

El target tiene 3 niveles, pero **la estratificación del split usa las 18 combinaciones `class = Fruit_Quality`** del manifest. Esto garantiza que train/val/test mantengan proporciones equilibradas por fruta y por calidad, evitando que el modelo entrene mayoritariamente con Pomegranate y se evalúe mayoritariamente con Lime.

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/annotations/manifest.csv")
y = df["quality"]                       # target = 3 clases
strat_key = df["class"]                 # estratificación = 18 combinaciones

train, test = train_test_split(df, test_size=0.15, stratify=strat_key, random_state=42)
train, val  = train_test_split(train, test_size=0.176, stratify=train["class"], random_state=42)
# 0.176 ≈ 0.15 / 0.85 para obtener 70/15/15 sobre el total
```

## Análisis de sesgo por especie (informe IEEE)

A partir del clasificador de 3 clases, se reporta:

- **Matriz de confusión 3×3 (target):** mide la calidad del clasificador.
- **Matriz de confusión 6×6 marginal por fruta:** cuenta cuántas imágenes de cada fruta caen en cada predicción. Idealmente la diagonal está cerca de la proporción real → el modelo no usa la especie como atajo.
- **Accuracy condicional por fruta:** `Acc(f) = #aciertos en imágenes de fruta f / total de fruta f`. Una disparidad alta sugiere que el modelo tiene sesgo por especie.

Esto responde la pregunta clave del informe: "¿el modelo aprende calidad real o pistas por especie?"

## Estimación de tamaño (salida secundaria)

Heurística basada en el área de la región frutal (segmentación por umbral en canal V de HSV + cierre morfológico). El umbral de tamaño se calibra **por especie**, porque "grande" no es comparable entre granada y lima:

```
tamaño(f) = {
  pequeño   si área < q₃₃(f)
  mediano   si q₃₃(f) ≤ área < q₆₆(f)
  grande    si área ≥ q₆₆(f)
}
```

donde `q₃₃(f)` y `q₆₆(f)` son los percentiles 33 y 66 del área para la fruta `f` en el conjunto de entrenamiento.

## Métricas reportadas

- **Globales (target = 3 clases):** Accuracy, Precision, Recall, F1 (por clase y macro).
- **Matriz 3×3 (target)** y **matriz 6×6 (especies, para sesgo).**
- **Curvas de aprendizaje** (loss y accuracy de train vs validation).
- **Comparativa tabular** SVM vs Random Forest vs CNN en F1-macro.
- **Análisis condicional** del rendimiento por especie.

Todas las figuras se guardan como **SVG** (requisito vectorial del informe IEEE) en `outputs/figures/`.
