#EJERCICIO1_DESCRIPTIVO
#IMPORTAR LIBRERIAS A UTILIZAR
import numpy as np
import pandas as pd
import matplotlib as mat
import seaborn as sb

#BASE DE DATOS EN CSV DEL PROYECTO
datos = pd.read_csv('data/TABLA_ACCIDENTES_24.XLSX - ACCIDENTES_24.csv')
print(datos)

#REVISAR ESTRUCTURA Y TIPO DE DATOS
print(datos.head(10))
print(datos.shape)
datos.info()
#Revisando los datos veo que tengo año y mes, realizo fecha
datos['FECHA'] = pd.to_datetime(
    datos['ANYO'].astype(str) + '-' + datos['MES'].astype(str) + '-01'
)
print(datos.info())
print(datos.dtypes)


#REVISION RESUMEN ESTRUCTURAL
print(f"Filas: {datos.shape[0]}")
print(f"Columnas: {datos.shape[1]}")
print(f"Memoria (MB): {datos.memory_usage(deep=True).sum() / 1024**2:.2f}")

#Revisión de nulos
nulos = datos.isnull().mean() * 100
nulos = nulos.sort_values(ascending=False)
print(nulos)

#Se detectan valores como nulos pero no lo son
datos['CONDICION_NIEBLA'] = datos['CONDICION_NIEBLA'].fillna(-1)
datos['CONDICION_VIENTO'] = datos['CONDICION_VIENTO'].replace('.', -1)
datos['CONDICION_VIENTO'] = datos['CONDICION_VIENTO'].fillna(-1)

#FUNCIONES : Las realizo para ser reutilizables en futuros proyectos
columnas_analisis = [
    'ID_ACCIDENTE',
    'TOTAL_VICTIMAS_24H',
    'TOTAL_MU24H',
    'TOTAL_HG24H',
    'TOTAL_VEHICULOS'
]

media = datos[columnas_analisis].mean()
mediana = datos[columnas_analisis].median()
moda = datos[columnas_analisis].mode().iloc[0]
varianza = ['TOTAL_VICTIMAS_24H'].var()

q1 = datos['TOTAL_VICTIMAS_24H'].quantile(0.25)
q2 = datos['TOTAL_VICTIMAS_24H'].quantile(0.50)
q3 = datos['TOTAL_VICTIMAS_24H'].quantile(0.75)

print(q1, q2, q3)

iqr = q3 - q1
print(iqr)

variable_objetivo = 'TOTAL_VICTIMAS_24H'

skewness = datos[variable_objetivo].skew()
curtosis = datos[variable_objetivo].kurtosis()

print("media:", media)
print("mediana", mediana)
print("moda:",moda)
print("varianza:", varianza)
print("q1:", q1)
print("q2:", q2)
print("q3:",q3)
print("iqr:",iqr)
print("Skewness:", skewness)
print("Curtosis:", curtosis)