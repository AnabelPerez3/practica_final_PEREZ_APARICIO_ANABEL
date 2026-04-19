"""
Ejercicio 4 — Análisis de Series Temporales
Asignatura: Estadística para Data Science
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# GENERACIÓN DE LA SERIE (NO MODIFICAR)
# ─────────────────────────────────────────────

def generar_serie_temporal(n_anios: int = 6, seed: int = 42) -> pd.Series:
    """
    Genera una serie temporal sintética diaria de n_anios años con:
- Tendencia lineal creciente
- Estacionalidad anual (seno con periodo 365 días)
- Ciclo de largo plazo (seno con periodo ~3 años)
- Ruido gaussiano

    *** NO MODIFICAR ESTA FUNCIÓN **¨
    """
    rng = np.random.default_rng(seed)
    n = 365 * n_anios
    t = np.arange(n)

    tendencia    = 0.005 * t
    estacional   = 10 * np.sin(2 * np.pi * t / 365)
    ciclo        = 5  * np.sin(2 * np.pi * t / (365 * 3))
    ruido        = rng.normal(0, 2, n)

    valores = 50 + tendencia + estacional + ciclo + ruido

    fechas = pd.date_range(start="2018-01-01", periods=n, freq="D")
    return pd.Series(valores, index=fechas, name="valor")


# ─────────────────────────────────────────────
# TAREA 6: VISUALIZACIÓN DE LA SERIE
# ─────────────────────────────────────────────

def visualizar_serie(serie: pd.Series) -> None:
    """
    Genera el gráfico de la serie temporal completa con título,
    etiquetas de ejes y cuadrícula. Guarda ej4_serie_original.png.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(serie.index, serie.values, color="#4C72B0", linewidth=0.8, alpha=0.9)
    ax.set_title("Serie temporal sintética — 6 años de datos diarios", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(alpha=0.3)
    # Anotación de tendencia
    ax.annotate(f"Inicio: {serie.values[0]:.1f}",xy=(serie.index[0], serie.values[0]),xytext=(50, 15), textcoords="offset points",fontsize=9, color="#DD4C4C",arrowprops=dict(arrowstyle="->", color="#DD4C4C"))
    ax.annotate(f"Fin: {serie.values[-1]:.1f}",xy=(serie.index[-1], serie.values[-1]),xytext=(-80, 15), textcoords="offset points",fontsize=9, color="#DD4C4C",arrowprops=dict(arrowstyle="->", color="#DD4C4C"))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ej4_serie_original.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej4_serie_original.png guardado")


# ─────────────────────────────────────────────
# TAREA 7: DESCOMPOSICIÓN
# ─────────────────────────────────────────────

def descomponer_serie(serie: pd.Series):
    """
    Aplica seasonal_decompose con model='additive' y period=365.
    Genera los 4 subgráficos (original, tendencia, estacional, residuo)
    y guarda ej4_descomposicion.png.
    """
    decomp = seasonal_decompose(serie, model="additive", period=365, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Descomposición aditiva — period=365",fontsize=13, fontweight="bold")

    componentes = [
        (serie,            "Serie original",  "#4C72B0"),
        (decomp.trend,     "Tendencia",       "#DD4C4C"),
        (decomp.seasonal,  "Estacionalidad",  "#2CA02C"),
        (decomp.resid,     "Residuo",         "#9467BD"),
    ]
    for ax, (comp, titulo, color) in zip(axes, componentes):
        ax.plot(comp.index, comp.values, color=color, linewidth=0.8)
        ax.set_ylabel(titulo, fontsize=10)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Fecha")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ej4_descomposicion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej4_descomposicion.png guardado")
    return decomp


# ─────────────────────────────────────────────
# TAREA 8: ANÁLISIS DEL RESIDUO
# ─────────────────────────────────────────────

def analizar_residuo(residuo: pd.Series) -> dict:
    """
    Calcula estadísticos descriptivos del residuo, aplica el test ADF
    (estacionariedad) y genera histograma con curva normal superpuesta,
    además del ACF y PACF.
    """
    r = residuo.dropna()

    print()
    print("=" * 60)
    print("ANÁLISIS DEL RESIDUO")
    print("=" * 60)

    # Estadísticos descriptivos
    media    = r.mean()
    std      = r.std()
    asim     = r.skew()
    curt     = r.kurt()
    print(f"  Media      : {media:.4f}")
    print(f"  Std        : {std:.4f}")
    print(f"  Asimetría  : {asim:.4f}")
    print(f"  Curtosis   : {curt:.4f}")

    # Test ADF (estacionariedad)
    adf_result = adfuller(r, autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    print()
    print("──── Test ADF (Augmented Dickey-Fuller) ────")
    print(f"  Estadístico ADF : {adf_stat:.4f}")
    print(f"  p-value         : {adf_pvalue:.6f}")
    print(f"  Conclusión      : {'ESTACIONARIO (p<0.05)' if adf_pvalue < 0.05 else 'NO estacionario (p≥0.05)'}")
    return resultados


# ─────────────────────────────────────────────
# GRÁFICO ACF / PACF
# ─────────────────────────────────────────────

def grafica_acf_pacf(residuo: pd.Series) -> None:
    """
    Genera los gráficos ACF y PACF del componente residuo
    y los guarda en ej4_acf_pacf.png.
    """
    r = residuo.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ACF y PACF del residuo", fontsize=13, fontweight="bold")

    plot_acf(r, lags=50, ax=axes[0], color="#4C72B0", alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=11)
    axes[0].set_xlabel("Lag")
    axes[0].grid(alpha=0.3)

    plot_pacf(r, lags=50, ax=axes[1], color="#4C72B0", alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=11)
    axes[1].set_xlabel("Lag")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ej4_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej4_acf_pacf.png guardado")


# ─────────────────────────────────────────────
# HISTOGRAMA DEL RESIDUO CON CURVA NORMAL
# ─────────────────────────────────────────────

def grafica_histograma_ruido(residuo: pd.Series) -> None:
    """
    Genera el histograma del residuo con la curva normal teórica
    superpuesta (ajustada a media y std del residuo).
    Guarda ej4_histograma_ruido.png.
    """
    r = residuo.dropna().values
    mu, sigma = r.mean(), r.std()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(r, bins=60, density=True, color="#4C72B0", alpha=0.7,
            edgecolor="white", label="Histograma residuo")

    # Curva normal teórica N(μ, σ²)
    x = np.linspace(r.min(), r.max(), 300)
    curva_normal = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, curva_normal, color="#DD4C4C", linewidth=2.5,
            label=f"Normal teórica N({mu:.2f}, {sigma:.2f}²)")

    ax.set_title(f"Histograma del residuo + curva normal teórica\n"f"μ={mu:.4f}  σ={sigma:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ej4_histograma_ruido.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ej4_histograma_ruido.png guardado")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n>>> EJERCICIO 4 — ANÁLISIS DE SERIES TEMPORALES <<<\n")

    # Generar serie (NO modificar esta llamada)
    serie = generar_serie_temporal(n_anios=6, seed=42)
    print(f"Serie generada: {len(serie)} observaciones diarias")
    print(f"Rango: {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"Min={serie.min():.2f}  Max={serie.max():.2f}  Media={serie.mean():.2f}")

    print()
    print("── Tarea 6: Visualización ──")
    visualizar_serie(serie)

    print()
    print("── Tarea 7: Descomposición ──")
    decomp = descomponer_serie(serie)

    print()
    print("── Tarea 8: Análisis del residuo ──")
    resultados = analizar_residuo(decomp.resid)
    grafica_acf_pacf(decomp.resid)
    grafica_histograma_ruido(decomp.resid)

    # Tendencia observada en el componente de tendencia
    tendencia = decomp.trend.dropna()
    pendiente = (tendencia.iloc[-1] - tendencia.iloc[0]) / len(tendencia)
    print()
    print(f"  Tendencia estimada: +{pendiente:.4f} unidades/día "f"(+{pendiente*365:.2f} unidades/año) → LINEAL CRECIENTE")
    print(f"  Amplitud estacional: ±{decomp.seasonal.max():.2f} unidades")

    print()
    print("Ejercicio 4 completado. Ficheros guardados en data/output/")
