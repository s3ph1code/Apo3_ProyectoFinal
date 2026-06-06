# Reporte EDA - Fase 1 CRISP-DM

> Generado desde `notebooks/01_comprension_datos.ipynb`.
> Proyecto: Clasificacion de calidad de frutas - APO3 ICESI 2026-1.

## 1. Resumen ejecutivo

- **Imagenes totales:** 9,515
- **Clases:** 18 (6 frutas x 3 calidades)
- **Fuentes:** {'recolectada': 9515}

## 2. Distribucion de clases

| Clase | n | % |
|---|---:|---:|
| Apple_Bad | 153 | 1.61% |
| Apple_Good | 267 | 2.81% |
| Apple_Regular | 890 | 9.35% |
| Banana_Bad | 684 | 7.19% |
| Banana_Good | 201 | 2.11% |
| Banana_Regular | 936 | 9.84% |
| Guava_Bad | 135 | 1.42% |
| Guava_Good | 200 | 2.10% |
| Guava_Regular | 813 | 8.54% |
| Lime_Bad | 152 | 1.60% |
| Lime_Good | 200 | 2.10% |
| Lime_Regular | 743 | 7.81% |
| Orange_Bad | 111 | 1.17% |
| Orange_Good | 206 | 2.17% |
| Orange_Regular | 510 | 5.36% |
| Pomegranate_Bad | 1009 | 10.60% |
| Pomegranate_Good | 1585 | 16.66% |
| Pomegranate_Regular | 720 | 7.57% |

**Imbalance Ratio (IR = n_max / n_min):** 14.28  
**Entropia normalizada (Hn = H / log K):** 0.903  
**Coeficiente de variacion (CV = sigma / mu):** 0.79

Clase mayoritaria: **Pomegranate_Good** (1,585).  
Clase minoritaria: **Orange_Bad** (111).

## 3. Dimensiones de imagen

| Estadistico | Width | Height |
|---|---:|---:|
| min | 13 | 12 |
| p25 | 103 | 100 |
| mediana | 495 | 434 |
| p75 | 3120 | 3120 |
| max | 4032 | 4032 |

Aspect ratio mediano: **1.00** (p5-p95: 0.56 - 1.85).

- Imagenes con lado < 100 px: **2562** (sospechosas: thumbnails o capturas corruptas).
- Imagenes con lado > 3000 px: **2531** (camara de celular en alta resolucion).

## 4. Hallazgos y decisiones para Fase 2 (preparacion)

1. **Desbalanceo severo (IR=14.3, Hn=0.90)** -> usar `class_weight='balanced'` en SVM/RF y CNN; augmentation mas agresiva en clases minoritarias (Orange_Bad: 111, Guava_Bad: 135, Lime_Bad: 152, Apple_Bad: 153).
2. **Resolucion heterogenea (13 px hasta 4032 px de lado)** -> resize fijo a 224x224 RGB normalizado a [0, 1]. Filtrar imagenes con lado < 64 px como outliers de calidad de captura.
3. **HSV como feature explicito** -> el espacio HSV separa especies por Hue y permite captar perdida de Saturation en frutas magulladas. Incluir histograma HSV de 96 bins en el vector de features de ML tradicional.
4. **Sesgo de clase (Pomegranate domina con 1585 en Good)** -> riesgo de que el modelo aprenda 'es granada' antes que 'es de calidad alta'. Reportar matriz de confusion 18x18 ademas de las marginales (3x3 calidad, 6x6 fruta).
5. **Sesgo etico (PI1)** -> el dataset es recoleccion estudiantil; condiciones de captura no controladas (iluminacion, angulo, fondo). El manifest registra `source` para trazabilidad. No se procesan rostros ni metadatos personales (EXIF GPS).
6. **Split estratificado por las 18 clases** (no solo por fruta o solo por calidad) con 70/15/15 train/val/test y `random_state=42` para reproducibilidad.

## 5. Figuras producidas

En `outputs/figures/eda/` (SVG vectorial para informe IEEE):

- `01_distribucion_18_clases.svg` - bar chart horizontal con conteo por clase.
- `02_heatmap_fruta_calidad.svg` - tabla cruzada 6x3 en heatmap.
- `03_porcentaje_vs_uniforme.svg` - proporcion de cada clase vs distribucion uniforme (1/18).
- `04_size_distributions.svg` - histogramas de width, height, aspect ratio.
- `05_muestras_grid.svg` - grid 6x3 de muestras visuales por fruta x calidad.
- `06_hsv_apple.svg`, `07_hsv_banana.svg` - histogramas HSV por calidad (se generan al ejecutar el notebook completo).
- `08_hue_sat_scatter.svg` - posicion cromatica promedio por clase (idem).

## 6. Proximos pasos (Fase 3 - Preparacion)

- Limpieza: filtrar imagenes con lado < 64 px; verificacion de duplicados con hash perceptual.
- Resize y normalizacion: 224x224 RGB en [0, 1] usando OpenCV.
- Split estratificado por `class` (las 18 clases): 70% train / 15% val / 15% test.
- Data augmentation con `albumentations`: HorizontalFlip, Rotate(+-25 deg), RandomBrightnessContrast. Mas iteraciones en clases minoritarias.
- Extraccion de features para ML tradicional (vector 141-D):
  - Histograma HSV: 96 bins (32 por canal).
  - LBP (Local Binary Patterns): 32 bins.
  - Momentos de Hu invariantes a rotacion/escala: 7.
  - Estadisticas por canal (mu, sigma para R, G, B): 6.
