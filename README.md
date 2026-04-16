# practica_final_-PEREZ_APARICIO_ANABEL-
Práctica final del módulo de Estadística Evolve. 
## 1. Selección del dataset
El dataset utilizado corresponde a accidentes de tráfico de la DGT del año 2024.
Cumple con los requisitos solicitados: 
- Tamaño: 42 columnas (superior al mínimo de 8 campos). 
- Peso: inferior a 15 MB tras la depuración de columnas.
### Variables categóricas:
- CARRETERA 
- TIPO_ACCIDENTE 
- TIPO_VIA 
- ZONA 
- CONDICION_ILUMINACION 
- CONDICION_METEO
### Variables numéricas continuas: 
- TOTAL_VICTIMAS_24H 
- TOTAL_MU24H 
- TOTAL_HG24H 
- TOTAL_VEHICULOS
### Variable objetivo (target): 
Se ha seleccionado **TOTAL_VICTIMAS_24H** como variable objetivo, ya que representa el número total de víctimas en accidentes y es adecuada para un problema de regresión.

## 1. Información del dataset - Resumen estructural
### Tamaño del dataset
El dataset está compuesto por 101.996 filas y 43 columnas. El tamaño en memoria es aproximadamente 41.67 MB.

### Tipos de datos
El conjunto de datos presenta una combinación de variables numéricas (int64 y float64), variables categóricas (object) y una variable temporal (datetime64). La mayoría de las variables son numéricas discretas relacionadas con características del accidente, mientras que variables como CARRETERA y KM son de tipo texto.

Además, se ha creado una variable adicional de tipo fecha (FECHA) a partir de las columnas ANYO y MES para facilitar el análisis temporal.

### Variables categóricas relevantes
Algunas variables meteorológicas han sido codificadas de forma categórica:

- CONDICION_NIEBLA:
  - 1 → Niebla ligera
  - 2 → Niebla intensa
  - 0 → No se sabe

- CONDICION_VIENTO:
  - 1 → Viento fuerte
  - 0 → No se sabe
  - NaN / "." → No se aprecia viento fuerte

Estas variables presentan una alta proporción de valores ausentes o “no detectados”, lo cual es coherente con la naturaleza de los fenómenos meteorológicos.

### Valores nulos
Se ha calculado el porcentaje de valores nulos por columna.

Las variables con mayor porcentaje de valores ausentes son:
- CONDICION_VIENTO (99.69%)
- CONDICION_NIEBLA (92.66%)
- KM (61.94%)

El resto de variables presentan valores completos o un porcentaje de nulos muy bajo.

### Interpretación de valores nulos
En el caso de las variables meteorológicas, los valores nulos o codificados como “no se sabe” pueden interpretarse como ausencia de información o condiciones no registradas en el momento del accidente.

Por ello, no se eliminan directamente estas variables, ya que pueden aportar información relevante en el análisis exploratorio posterior.

En su lugar:
- CONDICION_NIEBLA y CONDICION_VIENTO se mantienen como variables explicativas, interpretando los valores faltantes como ausencia de fenómeno relevante o falta de registro.
- KM presenta un porcentaje elevado de valores nulos, por lo que su tratamiento se evaluará en fases posteriores del análisis.
- TIPO_ACCIDENTE, con pocos valores nulos, puede ser imputada o eliminada sin impacto significativo.

Se ha realizado un tratamiento específico de las variables CONDICION_NIEBLA y CONDICION_VIENTO para diferenciar entre “no se sabe” y “sin registro”.
Se ha definido:
- 0 como “no se sabe”
- -1 como “sin información o no registrado”
- Valores positivos como condiciones meteorológicas reales observadas

Este enfoque permite diferenciar entre ausencia de fenómeno y falta de información en el registro.