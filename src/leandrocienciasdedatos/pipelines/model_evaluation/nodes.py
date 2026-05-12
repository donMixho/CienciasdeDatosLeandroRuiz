"""
Nodos del pipeline de evaluación de modelos (EV2).
Validación cruzada y comparación de métricas para
modelos de clasificación y regresión.
"""
import logging
import os
import glob
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from leandrocienciasdedatos.ml.data_preprocessing import (
    split_clasificacion,
    split_regresion,
)
from leandrocienciasdedatos.ml.model_evaluation import (
    evaluar_clasificacion,
    evaluar_regresion,
    validacion_cruzada,
)
from leandrocienciasdedatos.ml.data_preprocessing import FEATURES_CLASIFICACION, FEATURES_REGRESION

logger = logging.getLogger(__name__)
MODELS_PATH = "data/06_models"
PLOTS_PATH = "results/plots"
METRICS_PATH = "results/metrics"


def _cargar_modelos(sufijo: str) -> dict:
    """
    Carga modelos guardados en data/06_models/ según el sufijo (_clf o _reg).
    Retorna diccionario {nombre_modelo: pipeline_entrenado}.
    """
    modelos = {}
    patron = os.path.join(MODELS_PATH, f"*{sufijo}.pkl")
    archivos = glob.glob(patron)

    for ruta in archivos:
        nombre = os.path.basename(ruta).replace(sufijo + ".pkl", "")
        modelos[nombre] = joblib.load(ruta)
        logger.info(f"Modelo cargado: {nombre}")

    if not modelos:
        logger.warning(f"No se encontraron modelos con sufijo '{sufijo}'")
    return modelos


def evaluar_clasificacion_cv(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Evalúa modelos de clasificación con validación cruzada (CV=5).
    Genera gráfico comparativo de métricas.
    Retorna DataFrame con métricas promedio y desviación estándar.
    """
    cv = parameters.get("cv_folds", 5)
    X_train, X_test, y_train, y_test = split_clasificacion(dataset_ml)

    modelos = _cargar_modelos("_clf")
    if not modelos:
        return pd.DataFrame()

    # Validación cruzada
    X = dataset_ml[FEATURES_CLASIFICACION]
    y = dataset_ml["alto_desempeno"]
    df_cv = validacion_cruzada(modelos, X, y, tarea="clasificacion", cv=cv)

    # Métricas en test set
    df_test = evaluar_clasificacion(modelos, X_test, y_test)

    # Gráfico comparativo F1 score
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CV accuracy
    df_cv_sorted = df_cv.sort_values("accuracy_mean", ascending=True)
    axes[0].barh(df_cv_sorted["modelo"], df_cv_sorted["accuracy_mean"],
                xerr=df_cv_sorted["accuracy_std"], color="steelblue",
                alpha=0.8, capsize=5)
    axes[0].set_xlabel("Accuracy (CV)")
    axes[0].set_title(f"Accuracy Promedio (CV={cv})")
    axes[0].set_xlim(0, 1)

    # Test F1
    df_test_sorted = df_test.sort_values("f1_score", ascending=True)
    axes[1].barh(df_test_sorted["modelo"], df_test_sorted["f1_score"],
                color="darkorange", alpha=0.8)
    axes[1].set_xlabel("F1 Score (Test)")
    axes[1].set_title("F1 Score en Test Set")
    axes[1].set_xlim(0, 1)

    plt.suptitle("Comparación de Modelos — Clasificación", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "comparacion_clasificacion.png"), dpi=100)
    plt.close()

    # Guardar métricas
    os.makedirs(METRICS_PATH, exist_ok=True)
    df_test.to_csv(
        os.path.join(METRICS_PATH, "metricas_test_clasificacion.csv"),
        index=False, encoding="utf-8"
    )

    logger.info(f"Evaluación clasificación completada. CV={cv} folds.")
    return df_cv


def evaluar_regresion_cv(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Evalúa modelos de regresión con validación cruzada (CV=5).
    Genera gráfico comparativo de métricas.
    Retorna DataFrame con métricas promedio y desviación estándar.
    """
    cv = parameters.get("cv_folds", 5)
    X_train, X_test, y_train, y_test = split_regresion(dataset_ml)

    modelos = _cargar_modelos("_reg")
    if not modelos:
        return pd.DataFrame()

    # Validación cruzada
    X = dataset_ml[FEATURES_REGRESION]
    y = dataset_ml["avg_desempeno"]
    df_cv = validacion_cruzada(modelos, X, y, tarea="regresion", cv=cv)

    # Métricas en test set
    df_test = evaluar_regresion(modelos, X_test, y_test)

    # Gráfico comparativo R2
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_cv_sorted = df_cv.sort_values("r2_mean", ascending=True)
    axes[0].barh(df_cv_sorted["modelo"], df_cv_sorted["r2_mean"],
                xerr=df_cv_sorted["r2_std"], color="seagreen",
                alpha=0.8, capsize=5)
    axes[0].set_xlabel("R2 (CV)")
    axes[0].set_title(f"R2 Promedio (CV={cv})")

    df_test_sorted = df_test.sort_values("r2", ascending=True)
    axes[1].barh(df_test_sorted["modelo"], df_test_sorted["r2"],
                color="mediumpurple", alpha=0.8)
    axes[1].set_xlabel("R2 (Test)")
    axes[1].set_title("R2 en Test Set")

    plt.suptitle("Comparación de Modelos — Regresión", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "comparacion_regresion.png"), dpi=100)
    plt.close()

    os.makedirs(METRICS_PATH, exist_ok=True)
    df_test.to_csv(
        os.path.join(METRICS_PATH, "metricas_test_regresion.csv"),
        index=False, encoding="utf-8"
    )

    logger.info(f"Evaluación regresión completada. CV={cv} folds.")
    return df_cv