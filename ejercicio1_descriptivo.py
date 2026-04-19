"""
Ejercicio 1 - Análisis Estadístico Descriptivo
Dataset: Tabla de Accidentes de Tráfico 2024 (DGT)
Target: TOTAL_VICTIMAS_24H (número total de víctimas en las primeras 24h)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE RUTAS
# ─────────────────────────────────────────────
Datos = /Users/anabelperez/Downloads/Data Science Evolve/ESTADÍSTICA/TABLA_ACCIDENTES_24.XLSX - ACCIDENTES_24.csv
Output_datos = "data/output"
os.makedirs(Output_datos, exist_ok=True)

# ─────────────────────────────────────────────
# DICCIONARIOS DE ETIQUETAS
# ─────────────────────────────────────────────
ETIQ_TIPO_VIA = {
    1: "Autopista", 2: "Autovía", 3: "Vía multicalz.", 5: "Carretera conv.",6: "Vía urbana", 7: "Vía ciclista", 8: "Otro tipo", 9: "Se desconoce",12: "Variante", 14: "Carretera conv.2"
}
ETIQ_CONDICION_METEO = {
    1: "Despejado", 2: "Nublado", 3: "Lluvia débil", 4: "Lluvia fuerte",5: "Granizando", 6: "Nevando", 7: "Se desconoce", 999: "Sin especificar"
}
ETIQ_ZONA_AGRUPADA = {1: "Interurbana", 2: "Urbana"}
ETIQ_CONDICION_ILUM = {
    1: "Luz día", 2: "Amanecer/Atard. sin art.", 3: "Amanecer/Atard. con art.",4: "Sin natural, con art. enc.", 5: "Sin natural, art. apag.", 6: "Sin luz", 999: "Sin especificar"
}

# ─────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────

def cargar_datos(Datos:str) -> pd.DataFrame:
    """
    Carga y prepara el dataset de accidentes.
    """
    df = pd.read_csv(Datos)

    # Columnas categóricas como string legible
    df["TIPO_VIA_CAT"] = df["TIPO_VIA"].map(ETIQ_TIPO_VIA).fillna("Otro")
    df["CONDICION_METEO_CAT"] = df["CONDICION_METEO"].map(ETIQ_CONDICION_METEO).fillna("Otro")
    df["ZONA_AGRUPADA_CAT"] = df["ZONA_AGRUPADA"].map(ETIQ_ZONA_AGRUPADA).fillna("Otro")
    df["CONDICION_ILUM_CAT"] = df["CONDICION_ILUMINACION"].map(ETIQ_CONDICION_ILUM).fillna("Otro")

    return df
# ─────────────────────────────────────────────
# A) RESUMEN ESTRUCTURAL
# ─────────────────────────────────────────────

def resumen_estructural(df: pd.DataFrame) -> None:
    """
    Imprime el resumen estructural del dataset: filas, columnas, memoria,
    dtypes y porcentaje de nulos por columna.
    """
    print("=" * 60)
    print("A) RESUMEN ESTRUCTURAL")
    print("=" * 60)
    filas, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Filas       : {filas:,}")
    print(f"Columnas    : {cols}")
    print(f"Memoria     : {mem_mb:.2f} MB")
    print()
    print("Tipos de dato:")
    print(df.dtypes.to_string())
    print()
    nulos = (df.isnull().sum() / len(df) * 100).round(2)
    nulos_no_cero = nulos[nulos > 0]
    print("Porcentaje de nulos por columna (solo las que tienen):")
    print(nulos_no_cero.to_string())
    print()
    print("Decisión de tratamiento de nulos:")
    print("  - KM (61.95%): se descarta para el análisis -> muchos accidentes en zona urbana no tienen KM")
    print("  - CONDICION_NIEBLA (92.66%): se descarta -> casi toda la muestra no tiene niebla)")
    print("  - CONDICION_VIENTO (99.70%): se descarta -> prácticamente sin datos")
    print("  - TIPO_ACCIDENTE (0.03%): se imputa con la moda")
    print()

# ─────────────────────────────────────────────
# B) ESTADÍSTICOS DESCRIPTIVOS
# ─────────────────────────────────────────────

def estadisticos_descriptivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticos descriptivos completos de las variables numéricas
    relevantes e imprime métricas adicionales sobre la variable objetivo.
    """
    print("=" * 60)
    print("B) ESTADÍSTICOS DESCRIPTIVOS")
    print("=" * 60)

    cols_num = [
        "TOTAL_VICTIMAS_24H", "TOTAL_VEHICULOS", "TOTAL_MU24H",
        "TOTAL_HG24H", "TOTAL_HL24H", "HORA", "MES", "DIA_SEMANA"
    ]
    desc = df[cols_num].describe().T
    desc["varianza"]  = df[cols_num].var()
    desc["moda"]      = df[cols_num].mode().iloc[0]
    desc["skewness"]  = df[cols_num].skew()
    desc["curtosis"]  = df[cols_num].kurt()
    desc["IQR"]       = df[cols_num].quantile(0.75) - df[cols_num].quantile(0.25)

    print(desc.round(4).to_string())

    # Métricas extra sobre la variable objetivo
    target = df["TOTAL_VICTIMAS_24H"]
    q1, q3 = target.quantile(0.25), target.quantile(0.75)
    iqr = q3 - q1
    print()
    print("──── Variable objetivo: TOTAL_VICTIMAS_24H ────")
    print(f"  Media      : {target.mean():.4f}")
    print(f"  Mediana    : {target.median():.4f}")
    print(f"  Moda       : {target.mode()[0]}")
    print(f"  Std        : {target.std():.4f}")
    print(f"  Varianza   : {target.var():.4f}")
    print(f"  Min / Max  : {target.min()} / {target.max()}")
    print(f"  Q1 / Q3    : {q1} / {q3}")
    print(f"  IQR        : {iqr:.4f}")
    print(f"  Skewness   : {target.skew():.4f}  → distribución muy sesgada a la derecha")
    print(f"  Curtosis   : {target.kurt():.4f}  → leptocúrtica (colas pesadas)")

    # Guardar CSV
    desc.round(4).to_csv(f"{Output_datos}/ej1_descriptivo.csv")
    print(f"\nej1_descriptivo.csv guardado")
    return desc

# ─────────────────────────────────────────────
# C) DISTRIBUCIONES E HISTOGRAMAS
# ─────────────────────────────────────────────

def grafica_histogramas(df: pd.DataFrame) -> None:
    """
    Genera histogramas con KDE de todas las variables numéricas relevantes
    y los guarda en ej1_histogramas.png.
    """
    cols = [
        "TOTAL_VICTIMAS_24H", "TOTAL_VEHICULOS", "TOTAL_MU24H",
        "TOTAL_HG24H", "TOTAL_HL24H", "HORA", "MES", "DIA_SEMANA"
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Histogramas de variables numéricas - Accidentes 2024",fontsize=14, fontweight="bold", y=1.01)

    for ax, col in zip(axes.flat, cols):
        data = df[col].dropna()
        ax.hist(data, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
        # KDE superpuesta
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color="#DD4C4C", linewidth=2)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{Output_datos}/ej1_histogramas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej1_histogramas.png guardado")


def detectar_outliers(df: pd.DataFrame) -> None:
    """
    Detecta outliers en la variable objetivo mediante el método IQR
    e imprime un resumen.
    """
    print()
    print("──── Detección de outliers (método IQR) en TOTAL_VICTIMAS_24H ────")
    target = df["TOTAL_VICTIMAS_24H"]
    q1, q3 = target.quantile(0.25), target.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = df[(target < lim_inf) | (target > lim_sup)]
    pct = len(outliers) / len(df) * 100
    print(f"  Límite inferior: {lim_inf:.2f}   Límite superior: {lim_sup:.2f}")
    print(f"  Outliers detectados: {len(outliers):,} ({pct:.2f}%)")
    print("  Tratamiento: se conservan porque representan accidentes graves reales.En el modelo se aplicará transformación log1p al target para suavizar el sesgo.")

def grafica_boxplots(df: pd.DataFrame) -> None:
    """
    Genera boxplots de TOTAL_VICTIMAS_24H segmentados por cada variable
    categórica principal y los guarda en ej1_boxplots.png.
    """
    cat_cols = [
        ("ZONA_AGRUPADA_CAT","Zona"),
        ("CONDICION_METEO_CAT","Condición meteorológica"),
        ("CONDICION_ILUM_CAT","Condición iluminación"),
        ("TIPO_VIA_CAT","Tipo de vía"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Boxplots de TOTAL_VICTIMAS_24H por variable categórica",fontsize=14, fontweight="bold")

    for ax, (col, label) in zip(axes.flat, cat_cols):
        orden = (df.groupby(col)["TOTAL_VICTIMAS_24H"].median().sort_values(ascending=False).index)
        sns.boxplot(
            data=df, x=col, y="TOTAL_VICTIMAS_24H",
            order=orden, ax=ax,
            palette="Blues_d", fliersize=2, linewidth=0.8
        )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Total víctimas 24h")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{Output_datos}/ej1_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej1_boxplots.png guardado")


# ─────────────────────────────────────────────
# D) VARIABLES CATEGÓRICAS
# ─────────────────────────────────────────────

def grafica_categoricas(df: pd.DataFrame) -> None:
    """
    Genera gráficos de barras con frecuencias absolutas y relativas para
    las variables categóricas principales. Detecta desbalance.
    """
    print()
    print("=" * 60)
    print("D) VARIABLES CATEGÓRICAS")
    print("=" * 60)

    cat_info = [
        ("ZONA_AGRUPADA_CAT","Zona agrupada"),
        ("CONDICION_METEO_CAT","Condición meteorológica"),
        ("CONDICION_ILUM_CAT","Iluminación"),
        ("TIPO_VIA_CAT","Tipo de vía"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Frecuencia de variables categóricas – Accidentes 2024",fontsize=14, fontweight="bold")

    for ax, (col, titulo) in zip(axes.flat, cat_info):
        freq = df[col].value_counts()
        freq_rel = (freq / len(df) * 100).round(2)
        print(f"\n{titulo}:")
        for cat in freq.index:
            print(f"  {cat:35s} → {freq[cat]:6,}  ({freq_rel[cat]:.1f}%)")

        # Desbalance
        pct_max = freq_rel.iloc[0]
        if pct_max > 70:
            print(f"DESBALANCE: '{freq.index[0]}' domina con {pct_max:.1f}%")

        bars = ax.bar(freq.index, freq.values,color=sns.color_palette("muted", len(freq)),edgecolor="white")
        for bar, pct in zip(bars, freq_rel.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frecuencia absoluta")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{Output_datos}/ej1_categoricas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n ej1_categoricas.png guardado")


# ─────────────────────────────────────────────
# E) CORRELACIONES
# ─────────────────────────────────────────────

def grafica_correlaciones(df: pd.DataFrame) -> None:
    """
    Calcula la matriz de correlaciones de Pearson, genera el heatmap,
    identifica las 3 variables más correladas con el target y detecta
    multicolinealidad.
    """
    print()
    print("=" * 60)
    print("E) CORRELACIONES")
    print("=" * 60)

    cols_corr = [
        "TOTAL_VICTIMAS_24H", "TOTAL_VEHICULOS", "TOTAL_MU24H","TOTAL_HG24H", "TOTAL_HL24H", "HORA", "MES","DIA_SEMANA", "COD_PROVINCIA", "TIPO_VIA", "CONDICION_METEO","CONDICION_ILUMINACION", "CONDICION_FIRME"
    ]
    corr = df[cols_corr].corr(method="pearson")

    # Top-3 con el target
    corr_target = corr["TOTAL_VICTIMAS_24H"].drop("TOTAL_VICTIMAS_24H").abs().sort_values(ascending=False)
    print("Top 3 variables más correladas con TOTAL_VICTIMAS_24H:")
    for i, (var, val) in enumerate(corr_target.head(3).items()):
        print(f"  {i+1}. {var}: |r| = {val:.4f}")

    # Multicolinealidad
    print("\nPares con |r| > 0.9 (multicolinealidad):")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = abs(corr.iloc[i, j])
            if val > 0.9:
                print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: r = {corr.iloc[i, j]:.4f}")
                found = True
    if not found:
        print("  No se detecta multicolinealidad severa (|r| > 0.9) entre las predictoras analizadas.")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Matriz de correlaciones de Pearson – Accidentes 2024",fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{Output_datos}/ej1_heatmap_correlacion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej1_heatmap_correlacion.png guardado")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n EJERCICIO 1 — ANÁLISIS ESTADÍSTICO DESCRIPTIVO \n")

    df = cargar_datos(df = cargar_datos(Datos))

    # A) Resumen estructural
    resumen_estructural(df)

    # B) Estadísticos descriptivos
    estadisticos_descriptivos(df)

    # C) Distribuciones
    print()
    print("=" * 60)
    print("C) DISTRIBUCIONES Y OUTLIERS")
    print("=" * 60)
    grafica_histogramas(df)
    detectar_outliers(df)
    grafica_boxplots(df)

    # D) Variables categóricas
    grafica_categoricas(df)

    # E) Correlaciones
    grafica_correlaciones(df)

    print()
    print("Ejercicio 1 completado. Todos los ficheros en data/output/")
