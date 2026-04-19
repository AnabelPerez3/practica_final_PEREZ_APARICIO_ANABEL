"""
Ejercicio 3 — Regresión Lineal Múltiple en NumPy (OLS)
Asignatura: Estadística para Data Science
"""

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# IMPLEMENTACIÓN OLS
# ─────────────────────────────────────────────

def regresion_lineal_multiple(X_train: np.ndarray, y_train: np.ndarray,X_test: np.ndarray) -> tuple:
    """
    Calcula los coeficientes β mediante la solución analítica OLS
    y devuelve las predicciones sobre X_test.
    """
    n = X_train.shape[0]

    # Añadir columna de unos para el término independiente
    ones_train = np.ones((n, 1))
    X_train_aug = np.hstack([ones_train, X_train])

    # Usamos lstsq para mayor estabilidad numérica que la inversión directa
    XtX = X_train_aug.T @ X_train_aug          
    Xty = X_train_aug.T @ y_train             
    betas = np.linalg.lstsq(XtX, Xty, rcond=None)[0]  

    # Predicciones sobre test
    ones_test = np.ones((X_test.shape[0], 1))
    X_test_aug = np.hstack([ones_test, X_test])
    y_pred = X_test_aug @ betas              

    return y_pred, betas


def calcular_mae(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Mean Absolute Error (MAE).
    """
    return float(np.mean(np.abs(y_real - y_pred)))


def calcular_rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Root Mean Squared Error (RMSE).
    """
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def calcular_r2(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el coeficiente de determinación R².
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ─────────────────────────────────────────────
# GENERACIÓN DE DATOS SINTÉTICOS (NO MODIFICAR)
# ─────────────────────────────────────────────

def generar_datos_sinteticos(n: int = 1000, test_size: float = 0.2, seed: int = 42) -> tuple:
    """
    Genera datos sintéticos con coeficientes conocidos para validar
    la implementación OLS.
    """
    np.random.seed(seed)

    X = np.random.randn(n, 3)
    betas_reales = np.array([5.0, 2.0, -1.0, 0.5])  # β₀, β₁, β₂, β₃
    ruido = np.random.normal(0, 1.5, n)
    y = betas_reales[0] + X @ betas_reales[1:] + ruido

    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, betas_reales


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nEJERCICIO 3 — REGRESIÓN LINEAL MÚLTIPLE EN NUMPY (OLS)\n")

    # Generar datos sintéticos con semilla fija
    X_train, X_test, y_train, y_test, betas_reales = generar_datos_sinteticos(
        n=500, test_size=0.2, seed=42
    )

    print(f"Datos sintéticos generados: {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(f"Coeficientes REALES: β₀={betas_reales[0]}, β₁={betas_reales[1]}, "f"β₂={betas_reales[2]}, β₃={betas_reales[3]}")

    # Ajustar modelo OLS
    y_pred, betas_ajustados = regresion_lineal_multiple(X_train, y_train, X_test)

    print()
    print("──── Coeficientes ajustados vs reales ────")
    etiquetas = ["β₀ (intercepto)", "β₁", "β₂", "β₃"]
    for etiq, real, ajust in zip(etiquetas, betas_reales, betas_ajustados):
        print(f"  {etiq:18s}: real = {real:6.3f}  |  ajustado = {ajust:7.4f}  |  "f"error = {abs(real - ajust):.4f}")

    # Métricas
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    print()
    print("──── Métricas sobre test set sintético ────")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # Comparar con valores de referencia del profesor
    print()
    print("──── Comparación con valores de referencia del profesor ────")
    ref = {"β₀": 5.0, "β₁": 2.0, "β₂": -1.0, "β₃": 0.5,"MAE": 1.20, "RMSE": 1.50, "R2": 0.80}
    for k, v in ref.items():
        print(f"  {k:6s}: referencia ≈ {v:.2f}")

    # Gráfico Real vs Predicho
    grafica_real_vs_predicho(y_test, y_pred)

    print()
    print("Ejercicio 3 completado. Ficheros guardados en data/output/")
