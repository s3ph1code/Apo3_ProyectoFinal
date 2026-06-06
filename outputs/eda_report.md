# Reporte EDA - Fase 1 CRISP-DM

> Generado desde `notebooks/01_comprension_datos.ipynb`.
> Proyecto: Clasificacion universal de calidad de frutas - APO3 ICESI 2026-1.

## 1. Resumen ejecutivo

- **Target del modelo:** 3 clases universales de calidad (Good / Regular / Bad), agnosticas a la especie.
- **Imagenes totales:** 9,515
- **Fuentes:** {'recolectada': 9515}
- **Frutas representadas (metadata):** ['Apple', 'Banana', 'Guava', 'Lime', 'Orange', 'Pomegranate']

## 2. Distribucion del target (3 clases)

| Clase | n | % |
|---|---:|---:|
| Good | 2,659 | 27.95% |
| Regular | 4,612 | 48.47% |
| Bad | 2,244 | 23.58% |

**Imbalance Ratio (IR = n_max / n_min):** 2.06  
**Entropia normalizada (Hn):** 0.954  (1.0 = uniforme)
**Coeficiente de variacion (CV):** 0.40

El desbalanceo es leve. Clase mayoritaria: **Regular** (4,612). Clase minoritaria: **Bad** (2,244).

## 3. Composicion por especie (analisis de sesgo)

Aunque la especie no es target, su distribucion dentro de cada clase del target es critica para detectar sesgos:

| Calidad | Apple | Banana | Guava | Lime | Orange | Pomegranate |
|---|---:|---:|---:|---:|---:|---:|
| Good | 267 | 201 | 200 | 200 | 206 | 1585 |
| Regular | 890 | 936 | 813 | 743 | 510 | 720 |
| Bad | 153 | 684 | 135 | 152 | 111 | 1009 |

**Stats sobre las 18 combinaciones** (para estratificacion del split): IR=14.28, Hn=0.903, CV=0.79.

Observacion clave: **Pomegranate** representa el 60% de las imagenes Good (1585/2659) y el 45% de las Bad (1009/2244). Esto motiva la estratificacion del split por las 18 combinaciones para que el modelo no aprenda 'es granada' como atajo para 'es Good'.

## 4. Dimensiones de imagen

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

## 5. Hallazgos y decisiones para Fase 2 (preparacion)

1. **Desbalanceo leve (IR=2.06, Hn=0.954)** -> usar `class_weight='balanced'` por consistencia pero sin oversampling agresivo. La clase mayoritaria (Regular, 48.5%) duplica aproximadamente a la minoritaria (Bad, 23.6%).
2. **Estratificacion por las 18 combinaciones Fruit_Quality** -> garantiza balance fruta x calidad en train/val/test. Usar `stratify=df['class']` y target=`df['quality']`.
3. **Resolucion heterogenea (13 px hasta 4032 px de lado)** -> resize fijo a 224x224 RGB normalizado a [0, 1]. Filtrar imagenes con lado < 64 px como outliers de calidad de captura.
4. **HSV como feature explicito** -> aunque el target es agnostico a la especie, Hue/Saturation aportan informacion sobre calidad (frutas magulladas pierden Saturation). Incluir histograma HSV de 96 bins en el vector de features de ML tradicional.
5. **Analisis de sesgo obligatorio** -> evaluar matrices 3x3 (target) y 6x6 (por fruta). Reportar accuracy condicional por especie en el informe IEEE.
6. **Etica (PI1)** -> dataset es recoleccion estudiantil; condiciones de captura no controladas. Manifest registra `source` para trazabilidad. No se procesan rostros ni metadatos personales (EXIF GPS).

## 6. Figuras producidas

En `outputs/figures/eda/` (SVG vectorial para informe IEEE):

- `01_distribucion_18_clases.svg` - bar chart de las 18 combinaciones (para analisis de sesgo).
- `02_heatmap_fruta_calidad.svg` - tabla cruzada 6x3 en heatmap.
- `03_porcentaje_vs_uniforme.svg` - desviacion de la distribucion uniforme.
- `04_size_distributions.svg` - histogramas de width, height, aspect ratio.
- `05_muestras_grid.svg` - grid 6x3 de muestras visuales por fruta x calidad.
- `06_hsv_apple.svg`, `07_hsv_banana.svg` - histogramas HSV por calidad.
- `08_hue_sat_scatter.svg` - posicion cromatica promedio por clase.
- **`09_distribucion_3_clases.svg`** - distribucion del target (3 clases) + composicion por especie.

## 7. Proximos pasos (Fase 3 - Preparacion)

- Limpieza: filtrar imagenes con lado < 64 px; verificacion de duplicados con hash perceptual.
- Resize y normalizacion: 224x224 RGB en [0, 1] usando OpenCV.
- Split estratificado por `class` (18) con target=`quality` (3): 70% train / 15% val / 15% test.
- Data augmentation con `albumentations`: HorizontalFlip, Rotate(+-25 deg), RandomBrightnessContrast.
- Extraccion de features para ML tradicional (vector 141-D):
  - Histograma HSV: 96 bins (32 por canal).
  - LBP (Local Binary Patterns): 32 bins.
  - Momentos de Hu invariantes a rotacion/escala: 7.
  - Estadisticas por canal (mu, sigma para R, G, B): 6.
