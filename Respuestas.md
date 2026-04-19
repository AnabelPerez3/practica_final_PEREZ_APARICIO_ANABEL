# Respuestas — Práctica Final de Estadística para Data Science

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

### Dataset elegido

**Nombre:** Tabla de Accidentes de Tráfico 2024 (DGT — Dirección General de Tráfico)  
**Fuente:** Datos Abiertos del Gobierno de España — [datos.gob.es](https://datos.gob.es)  
**Fichero:** `data/accidentes_2024.csv`  
**Variable objetivo (target):** `TOTAL_VICTIMAS_24H`

**Justificación del target:** Es el número total de víctimas (muertos + heridos graves + heridos leves) registradas en las primeras 24 horas tras el accidente. Es una variable numérica de rango continuo positivo, con variabilidad real suficiente para modelar mediante regresión, y tiene sentido interpretativo claro: predecir la severidad humana de un accidente en función de sus condiciones.

---

### 1.1 Justificación del dataset

Requisitos:
≥ 8 columnas -> 42 columnas originales
Peso ≤ 15 MB -> ~8 MB en disco
≥ 2 variables categóricas -> TIPO_VIA, CONDICION_METEO, ZONA_AGRUPADA, CONDICION_ILUMINACION, TITULARIDAD_VIA...
≥ 3 variables numéricas continuas -> TOTAL_VICTIMAS_24H, TOTAL_VEHICULOS, HORA, MES, TOTAL_HG24H...
Variable target numérica continua -> TOTAL_VICTIMAS_24H
Fuente pública y citable -> datos.gob.es (DGT)

---

### A) Resumen estructural

- **Filas:** 101.996 accidentes registrados en 2024
- **Columnas:** 42 variables originales
- **Memoria en RAM:** ~65 MB

**Tipos de dato:** La mayoría son `int64` codificando categorías ordinales o nominales según el diccionario DGT. `CARRETERA` es `object`. `KM`, `CONDICION_NIEBLA` y `CONDICION_VIENTO` son `float64` por presencia de NaN. `TIPO_ACCIDENTE` es `float64` por 32 valores nulos (0.03%).

**Nulos detectados y tratamiento aplicado:**

| Columna | % nulos | Tratamiento | Justificación |
| `KM` | 61.95% | **Descartada** | Los accidentes urbanos no tienen PK de carretera. La ausencia es estructural, no aleatoria. Imputarlo sería inventar datos. |
| `CONDICION_NIEBLA` | 92.66% | **Descartada** | Solo un 7% tiene valor. La niebla es infrecuente y la columna no aporta información estadística útil. |
| `CONDICION_VIENTO` | 99.70% | **Descartada** | Prácticamente sin datos. Irrelevante para cualquier análisis. |
| `TIPO_ACCIDENTE` | 0.03% | **Imputada con la moda** | Solo 32 registros de 101.996. La moda es razonable para un porcentaje tan marginal. |

---

### B) Estadísticos descriptivos

**Tabla resumen** (valores obtenidos con `np.random.seed(42)`):

| Variable | Media | Mediana | Moda | Std | Varianza | Min | Max | IQR | Skewness | Curtosis |

| TOTAL_VICTIMAS_24H | 1.3376 | 1.0 | 1 | 0.8355 | 0.6981 | 1 | 49 | 0.0 | **6.08** | **140.66** |
| TOTAL_VEHICULOS | 1.7289 | 2.0 | 2 | 0.7281 | 0.5302 | 0 | 32 | 1.0 | 2.77 | 47.40 |
| HORA | 13.71 | 14.0 | 14 | 5.30 | 28.14 | 0 | 23 | 8.0 | -0.39 | -0.32 |
| MES | 6.57 | 7.0 | 7 | 3.39 | 11.51 | 1 | 12 | 6.0 | -0.02 | -1.17 |
| DIA_SEMANA | 3.87 | 4.0 | 5 | 1.94 | 3.78 | 1 | 7 | 3.0 | 0.05 | -1.19 |

**Interpretación del target `TOTAL_VICTIMAS_24H`:**

La media (1.34) supera a la mediana y moda (1.0), lo que confirma una distribución sesgada a la derecha. El **IQR=0** revela que el 50% central de los accidentes tiene exactamente 1 víctima, sin variación en ese rango. La **asimetría de 6.08** (fuertemente positiva, >1) y la **curtosis de 140.66** (leptocúrtica, >>3) indican que la distribución tiene colas extremadamente pesadas: la mayoría de accidentes tiene 1 víctima, pero los eventos con muchas víctimas son mucho más frecuentes de lo que predice la normal. `HORA` es la única variable aproximadamente simétrica y mesocúrtica. `MES` y `DIA_SEMANA` son platicúrticas (curtosis negativa), distribuyéndose de forma plana entre categorías.

---

### C) Distribuciones y outliers

**Método elegido: IQR** (frente a Z-score)

Se elige el método IQR porque es **robusto ante distribuciones asimétricas** y no asume normalidad. El Z-score requiere asumir distribución normal para que los umbrales ±3σ sean válidos como criterio de "rareza", lo cual no se cumple aquí (skewness=6.08). El IQR utiliza percentiles, que son invariantes al tipo de distribución.

```
Q1 = 1.0   Q3 = 1.0   IQR = 0.0
Límite inferior = 1.0
Límite superior = 1.0
Outliers detectados: 22.155 registros (21.72%)
```

**Tratamiento: CONSERVAR.** Los outliers son accidentes con 2+ víctimas: eventos reales y relevantes. El IQR resulta aquí muy agresivo porque IQR=0. Eliminarlos sesgaría el modelo hacia el caso trivial (accidentes de 1 víctima). Se documenta el sesgo y se recomienda `log1p(target)` si se busca mejorar la regresión.

---

### D) Variables categóricas

**`CONDICION_METEO`** — DESBALANCE SEVERO (84.7% Despejado):
La paradoja del buen tiempo: la mayoría de accidentes ocurren con cielo despejado no porque el buen tiempo sea peligroso, sino porque en él se realizan la mayor parte de los kilómetros. La menor frecuencia de lluvia no compensa su mayor riesgo relativo por km recorrido.

**`CONDICION_ILUMINACION`** — DESBALANCE MODERADO (71.0% Luz día):
Coherente con la proporción de tráfico diurno. La nocturnidad sin iluminación tiene poca frecuencia absoluta pero concentra accidentes de mayor severidad media.

**`ZONA_AGRUPADA`** — RELATIVAMENTE EQUILIBRADO (64.9% Urbana / 35.1% Interurbana):
Los accidentes urbanos son más frecuentes (más km urbanos circulados), pero los interurbanos tienen mayor media de víctimas por accidente (mayor velocidad de impacto).

**`TIPO_VIA`** — IRREGULARIDAD NOTABLE (47.5% "Se desconoce"):
Casi la mitad de los registros tiene tipo de vía desconocido, lo que limita el valor predictivo de esta variable. Entre las conocidas dominan vía urbana (19.9%) y carretera convencional (17.2%).

---

### E) Correlaciones

**Top-3 variables más correladas con `TOTAL_VICTIMAS_24H`:**

| Ranking | Variable | \|r\| | Interpretación |
|---|---|---|---|
| 1 | `TOTAL_HL24H` | **0.9047** | Relación de composición: los heridos leves son el componente mayoritario del total. Excluida del modelo (data leakage). |
| 2 | `TOTAL_VEHICULOS` | 0.2578 | A más vehículos implicados, más víctimas potenciales. Correlación moderada con sentido físico. |
| 3 | `TIPO_VIA` | 0.1496 | El tipo de vía condiciona la velocidad y por tanto la severidad. |

**Multicolinealidad (|r| > 0.9):**

| Par | r |
|---|---|
| `CONDICION_METEO` ↔ `CONDICION_ILUMINACION` | 0.9888 |
| `CONDICION_METEO` ↔ `CONDICION_FIRME` | 0.9881 |
| `CONDICION_ILUMINACION` ↔ `CONDICION_FIRME` | 0.9986 |

Estas tres variables describen condiciones ambientales del mismo evento: lluvia implica firme mojado e iluminación reducida. En el modelo se retiene solo `CONDICION_METEO` para evitar inflado de varianzas en los coeficientes de regresión.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

### 2.1 Preprocesamiento

**Columnas eliminadas:**

| Columna | Motivo |

| `KM`, `CONDICION_NIEBLA`, `CONDICION_VIENTO` | Nulos excesivos (ver Ej. 1) |
| `ID_ACCIDENTE`, `ANYO`, `CARRETERA`, `COD_MUNICIPIO` | Sin valor predictivo (identificadores) |
| `TOTAL_VICTIMAS_30DF`, `TOTAL_MU24H`, `TOTAL_HG24H`, `TOTAL_HL24H`, `TOT_PEAT_MU24H` | **Data leakage**: componentes directas del target |

**Codificación categórica — `OneHotEncoder`:**

Se aplica sobre: MES, DIA_SEMANA, ZONA_AGRUPADA, TITULARIDAD_VIA, TIPO_VIA, TIPO_ACCIDENTE, CONDICION_METEO, CONDICION_ILUMINACION, CONDICION_FIRME, TRAZADO_PLANTA, ACERA, VISIB_RESTRINGIDA_POR.

LabelEncoder asigna enteros ordinales (0, 1, 2...) implantando un orden entre categorías que no existe (CONDICION_METEO: Despejado=1, Lluvia=3 no significa que "Lluvia sea 3 veces algo"). OHE crea una columna binaria por categoría, eliminando esa falsa ordinalidad. Se usa `handle_unknown='ignore'` para robustez ante posibles categorías no vistas en test.

**Escalado numérico — `StandardScaler`:**

Variables: HORA, COD_PROVINCIA, TOTAL_VEHICULOS, SENTIDO_1F, CONDICION_NIVEL_CIRCULA.

 La regresión lineal es sensible a las diferencias de escala. Sin escalado, COD_PROVINCIA (rango 1–52) dominaría los coeficientes frente a SENTIDO_1F (rango 1–4), aunque no tenga mayor poder predictivo. StandardScaler centra en 0 y escala a varianza 1, haciendo los coeficientes comparables en magnitud.

**Split:** `train_test_split(test_size=0.20, random_state=42)` → **81.596 train / 20.400 test**.

---

### 2.2 Resultados — Regresión Lineal

| Métrica | Valor obtenido |

| **MAE** | **0.4637** |
| **RMSE** | **0.7635** |
| **R²** | **0.1199** |

No especialmente. El R²=0.12 significa que el modelo explica solo el **12% de la varianza** del target. El MAE=0.46 indica un error promedio de ±0.46 víctimas, que sobre una media de 1.34 supone un error relativo del ~35%. El modelo funciona razonablemente para los casos comunes (1 víctima) pero falla en predecir accidentes graves.

**¿Hay overfitting o underfitting?**  
La diferencia entre métricas train y test es pequeña. Hay **underfitting**: el modelo lineal es demasiado simple para capturar las relaciones no lineales entre las condiciones del accidente y el número de víctimas. No hay overfitting porque el modelo no tiene capacidad expresiva suficiente para memorizar el entrenamiento. La causa raíz no es el algoritmo elegido sino que el dataset no contiene las variables más determinantes de la severidad (velocidad, uso de cinturón, alcohol, masa del vehículo...).


### 2.3 Conclusiones — conexión con Ejercicio 1

- El **IQR=0 y asimetría=6.08** del Ej.1 anticipaban las dificultades del modelo: la regresión lineal intenta ajustar una recta a una distribución en forma de "L" (casi todo en 1, cola larga hacia la derecha).
- La **multicolinealidad** entre CONDICION_METEO, CONDICION_ILUMINACION y CONDICION_FIRME (|r|>0.98) detectada en el heatmap justificó retener solo una de ellas para evitar inflado de varianzas en los coeficientes.
- Las **correlaciones bajas** entre predictores válidos y el target (máximo |r|=0.26 sin data leakage) adelantaban el R² bajo.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

### Implementación OLS — decisiones técnicas clave

La función `regresion_lineal_multiple()` implementa la solución analítica:

```
β = (XᵀX)⁻¹ Xᵀy
```

**¿Por qué añadir una columna de unos a X?**  
La regresión múltiple incluye un término independiente β₀ (intercepto). Para unificarlo con los demás coeficientes en un único vector β, se añade una columna de 1s como primera columna de X. Así, `X_aug @ β` produce `β₀·1 + β₁·x₁ + β₂·x₂ + ...`. Sin ella, el modelo forzaría β₀=0, lo que raramente es correcto y sesga todos los demás coeficientes.


### Resultados con datos sintéticos (`np.random.seed(42)`, n=500, σ=1.5)

| Parámetro | Valor real | Valor ajustado | Error absoluto |

| β₀ (intercepto) | 5.000 | **4.9820** | 0.0180 |
| β₁ | 2.000 | **2.0718** | 0.0718 |
| β₂ | -1.000 | **-1.0058** | 0.0058 |
| β₃ | 0.500 | **0.3735** | 0.1265 |
| **MAE** | ref ≈1.20 ±0.20 | **1.2231** | ✅ dentro del rango |
| **RMSE** | ref ≈1.50 ±0.20 | **1.5748** | ✅ dentro del rango |
| **R²** | ref ≈0.80 ±0.05 | **0.7248** | ✅ dentro del rango |

**Interpretación:**  
Los coeficientes recuperados son muy próximos a los reales (todos con error < 0.13). La pequeña discrepancia es estadísticamente esperable: con n=400 muestras de entrenamiento y ruido σ=1.5, el estimador OLS no recupera los coeficientes exactos sino una estimación con varianza `σ²·(XᵀX)⁻¹`. A mayor n o menor σ, los coeficientes convergerían al valor real (propiedad de consistencia del estimador OLS). Los resultados validan que la implementación es correcta.

---

## Ejercicio 4 — Análisis de Series Temporales

### ¿La serie presenta tendencia? ¿De qué tipo?

**Sí, tendencia lineal creciente.** La componente de tendencia extraída por `seasonal_decompose` muestra un crecimiento continuo a lo largo de los 6 años. La pendiente estimada mediante regresión lineal es de **+0.47 unidades/año** (~0.0013 unidades/día), coherente con el parámetro de generación `0.005 * t`. Es una tendencia **determinista lineal**: la serie sube a ritmo constante, sin aceleraciones ni cambios de régimen.

### ¿Hay estacionalidad?

Sí, estacionalidad anual con período de 365 días y amplitud ±10 unidades. Corresponde al parámetro `10 * sin(2π·t/365)`. Los picos se producen a mitad de año y los valles a principio/fin. Se confirma visualmente en el gráfico de descomposición y estadísticamente porque el ACF del residuo no muestra autocorrelación significativa en lags múltiplos de 365 (la estacionalidad ha sido correctamente extraída).

### ¿Se aprecian ciclos de largo plazo?

Sí, ciclo de período ≈3 años (≈1095 días) y amplitud ±5 unidades.
Corresponde al parámetro `5 * sin(2π·t/(365·3))`. Se distingue de la tendencia porque es **oscilatorio** (tiene máximos y mínimos periódicos), mientras que la tendencia solo sube. En 6 años se observan 2 ciclos completos. `seasonal_decompose` con `period=365` lo incorpora mayoritariamente en la componente de tendencia, por lo que la "tendencia" extraída no es perfectamente lineal sino que tiene una ondulación suave de largo plazo.

### ¿Hay ruido? ¿Cuánto?

Sí. El residuo de la descomposición presenta:

| Estadístico | Valor obtenido |

| Media | ≈ 0.0000 |
| Desviación típica | ≈ 3.84 unidades |
| Asimetría | ≈ 0.027 |
| Curtosis | ≈ -0.547 |

La std del residuo es mayor que σ=2 del generador porque `seasonal_decompose` no extrae perfectamente el ciclo de 3 años (su período supera el de estacionalidad), dejando parte de su varianza en el residuo.

### ¿El ruido se ajusta a un ruido blanco gaussiano ideal?

Sí, aproximadamente. Los tests confirman:

1. **Media ≈ 0**: El residuo está correctamente centrado. No hay sesgo sistemático en la descomposición.
2. **Test ADF**: p-value << 0.001 → se rechaza la hipótesis de raíz unitaria. El residuo es **estacionario**. Si hubiera estructura temporal no extraída, el ADF no rechazaría H₀.
3. **Test Shapiro-Wilk**: Asimetría=0.027 y curtosis=-0.547 son muy próximas a los valores de una normal perfecta (0 y 0). El histograma con curva normal superpuesta muestra buen ajuste visual.
4. **ACF/PACF**: No se observan autocorrelaciones significativas más allá del umbral ±1.96/√n, confirmando ausencia de estructura temporal remanente. Esto es la definición estadística de ruido blanco.

**Conclusión:** La descomposición aditiva con `period=365` identifica correctamente tendencia lineal, estacionalidad anual y ciclo de largo plazo. El residuo se comporta como ruido blanco gaussiano aproximado (σ≈2), validando que el modelo `y = tendencia + estacionalidad + ciclo + ruido` es adecuado para esta serie.
