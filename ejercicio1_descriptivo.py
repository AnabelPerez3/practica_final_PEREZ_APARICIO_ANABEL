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

def calcular_media(df, columnas_analisis):
    media = df[columnas_analisis].mean()
    return {"media": media}

def calcular_mediana(df, columnas_analisis):
    mediana = df[columnas_analisis].median()
    return {"mediana": mediana}

def calcular_moda(df, columnas_analisis):
    moda = df[columnas_analisis].mode().iloc[0]
    return {"moda": moda}

    
print("Media:\n", calcular_media(datos, columnas_analisis))
print("\nMediana:\n", calcular_mediana(datos, columnas_analisis))
print("\nModa:\n", calcular_moda(datos, columnas_analisis))

def calcular_varianza(df,columnas_analisis):
    return df[columnas_analisis].var()

print("Varianza:\n", calcular_varianza(datos, columnas_analisis))

