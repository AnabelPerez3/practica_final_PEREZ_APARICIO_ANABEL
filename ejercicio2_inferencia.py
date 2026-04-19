"""
Ejercicio 2 — Inferencia con Scikit-Learn
Target: TOTAL_VICTIMAS_24H
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "data/accidentes_2024.csv"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────

def cargar_y_preparar(path: str) -> pd.DataFrame:
    """
    Carga el CSV, imputa el único campo con nulos relevantes y
    descarta columnas de muy alto porcentaje de nulos o sin valor
    predictivo para el target.
    Dataset limpio y listo para modelar.
    """
    df = pd.read_csv(path, sep=None, engine="python")

    # Imputar TIPO_ACCIDENTE (0.03% nulos) → moda
    df["TIPO_ACCIDENTE"] = df["TIPO_ACCIDENTE"].fillna(df["TIPO_ACCIDENTE"].mode()[0])

    # Descartar columnas con >50% nulos o sin aporte predictivo
    cols_drop = [
        "KM", "CONDICION_NIEBLA", "CONDICION_VIENTO",
        "ID_ACCIDENTE", "ANYO", "CARRETERA", "COD_MUNICIPIO",
        # Columnas target derivadas (fugan información)
        "TOTAL_VICTIMAS_30DF", "TOTAL_MU24H", "TOTAL_HG24H",
        "TOTAL_HL24H", "TOT_PEAT_MU24H"
    ]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    return df


# ─────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────

def preprocesar(df: pd.DataFrame):
    """
    Aplica preprocesamiento: define features y target, construye
    ColumnTransformer con OneHotEncoder para categóricas y StandardScaler
    para numéricas y realiza el split 80/20.
    """
    TARGET = "TOTAL_VICTIMAS_24H"

    # Columnas categóricas (código numérico pero semántica nominal)
    cat_features = [
        "MES", "DIA_SEMANA", "ZONA_AGRUPADA", "TITULARIDAD_VIA",
        "TIPO_VIA", "TIPO_ACCIDENTE", "CONDICION_METEO",
        "CONDICION_ILUMINACION", "CONDICION_FIRME", "TRAZADO_PLANTA",
        "ACERA", "VISIB_RESTRINGIDA_POR"
    ]
    # Columnas numéricas continuas
    num_features = ["HORA", "COD_PROVINCIA", "TOTAL_VEHICULOS",
                    "SENTIDO_1F", "CONDICION_NIVEL_CIRCULA"]

    # Filtrar solo columnas existentes
    cat_features = [c for c in cat_features if c in df.columns]
    num_features = [c for c in num_features if c in df.columns]

    X = df[cat_features + num_features]
    y = df[TARGET]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")
    print(f"  Features numéricas : {num_features}")
    print(f"  Features categóricas (OHE): {cat_features}")

    return X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features


# ─────────────────────────────────────────────
# REGRESIÓN LINEAL
# ─────────────────────────────────────────────

def entrenar_regresion_lineal(X_train, X_test, y_train, y_test, preprocessor):
    """
    Construye un pipeline con preprocesador + LinearRegression,
    entrena sobre train y evalúa sobre test calculando MAE, RMSE y R².
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metricas = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print()
    print("──── Métricas Regresión Lineal ────")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # Guardar métricas en fichero de texto
    with open(f"{OUTPUT_DIR}/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
        f.write("=== MÉTRICAS — REGRESIÓN LINEAL (LinearRegression) ===\n\n")
        f.write(f"Conjunto de test: 20% ({len(y_test):,} muestras)\n\n")
        f.write(f"MAE  (Mean Absolute Error)   = {mae:.6f}\n")
        f.write(f"RMSE (Root Mean Sq. Error)   = {rmse:.6f}\n")
        f.write(f"R²   (Coef. Determinación)   = {r2:.6f}\n\n")
        f.write("Interpretación:\n")
        f.write("  El MAE indica el error promedio en número de víctimas.El R² mide la proporción de varianza explicada por el modelo.Un R² bajo sugiere que el número de víctimas tiene alta aleatoriedad.inherente y depende de factores no recogidos en el dataset (velocidad, uso de cinturón, alcohol, etc.).\n")
    print("ej2_metricas_regresion.txt guardado")

    return pipeline, y_pred, metricas


# ─────────────────────────────────────────────
# GRÁFICO DE RESIDUOS
# ─────────────────────────────────────────────

def grafica_residuos(y_test, y_pred):
    """
    Genera el gráfico de residuos (predichos en X, residuos en Y)
    con línea de referencia en 0 y guarda ej2_residuos.png.
    """
    residuos = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Análisis de residuos — Regresión Lineal", fontsize=13, fontweight="bold")

    # Plot principal: predichos vs residuos
    axes[0].scatter(y_pred, residuos, alpha=0.2, s=5, color="#4C72B0")
    axes[0].axhline(0, color="#DD4C4C", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Valores predichos")
    axes[0].set_ylabel("Residuos")
    axes[0].set_title("Predichos vs Residuos")
    axes[0].grid(alpha=0.3)

    # Histograma de residuos
    axes[1].hist(residuos, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
    from scipy import stats
    kde_x = np.linspace(residuos.min(), residuos.max(), 300)
    kde = stats.gaussian_kde(residuos)
    axes[1].plot(kde_x, kde(kde_x), color="#DD4C4C", linewidth=2)
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Distribución de residuos")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ej2_residuos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] ej2_residuos.png guardado")


# ─────────────────────────────────────────────
# ANÁLISIS DE COEFICIENTES
# ─────────────────────────────────────────────

def analizar_coeficientes(pipeline, num_features, cat_features):
    """
    Extrae y muestra los coeficientes del modelo lineal por variable,
    identificando las features más influyentes.
    """
    modelo = pipeline.named_steps["regressor"]
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_features).tolist()
    feature_names = num_features + cat_names

    coefs = pd.Series(modelo.coef_, index=feature_names).abs().sort_values(ascending=False)
    top10 = coefs.head(10)

    print()
    print("──── Top-10 variables más influyentes (|coeficiente|) ────")
    for feat, val in top10.items():
        print(f"  {feat:45s}: {val:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n>>> EJERCICIO 2 — INFERENCIA CON SCIKIT-LEARN <<<\n")

    print("── Carga y preparación ──")
    df = cargar_y_preparar(DATA_PATH)

    print("\n── Preprocesamiento ──")
    X_train, X_test, y_train, y_test, preprocessor, num_feat, cat_feat = preprocesar(df)

    print("\n── Regresión Lineal ──")
    pipeline, y_pred, metricas = entrenar_regresion_lineal(
        X_train, X_test, y_train, y_test, preprocessor
    )

    graficar_residuos(y_test, y_pred)
    analizar_coeficientes(pipeline, num_feat, cat_feat)

    print()
    print(">>> Ejercicio 2 completado. Ficheros guardados en data/output/ <<<")
