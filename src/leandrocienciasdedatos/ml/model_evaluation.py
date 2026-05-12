"""
Funciones de evaluación y comparación de modelos.
Métricas para clasificación y regresión con validación cruzada.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


def evaluar_clasificacion(modelos: dict, X_test: pd.DataFrame,
    y_test: pd.Series) -> pd.DataFrame:
    """
    Evalúa modelos de clasificación con múltiples métricas.
    Retorna DataFrame comparativo con accuracy, precision, recall, F1.
    """
    resultados = []
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        resultados.append({
            "modelo": nombre,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    df_resultados = pd.DataFrame(resultados).sort_values("f1_score", ascending=False)
    logger.info("Evaluación de clasificación completada.")
    return df_resultados


def evaluar_regresion(modelos: dict, X_test: pd.DataFrame,
    y_test: pd.Series) -> pd.DataFrame:
    """
    Evalúa modelos de regresión con múltiples métricas.
    Retorna DataFrame comparativo con RMSE, MAE, R2.
    """
    resultados = []
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        resultados.append({
            "modelo": nombre,
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        })
    df_resultados = pd.DataFrame(resultados).sort_values("r2", ascending=False)
    logger.info("Evaluación de regresión completada.")
    return df_resultados


def validacion_cruzada(modelos: dict, X: pd.DataFrame,
    y: pd.Series, tarea: str = "clasificacion",
    cv: int = 5) -> pd.DataFrame:
    """
    Aplica validación cruzada (CV=5) a todos los modelos.
    tarea: 'clasificacion' o 'regresion'
    """
    if tarea == "clasificacion":
        scoring = ["accuracy", "f1", "precision", "recall"]
    else:
        scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

    resultados = []
    for nombre, modelo in modelos.items():
        try:
            scores = cross_validate(modelo, X, y, cv=cv, scoring=scoring)
            fila = {"modelo": nombre}
            for metrica in scoring:
                vals = scores[f"test_{metrica}"]
                fila[f"{metrica}_mean"] = round(vals.mean(), 4)
                fila[f"{metrica}_std"] = round(vals.std(), 4)
            resultados.append(fila)
            logger.info(f"CV completado: {nombre}")
        except Exception as e:
            logger.error(f"Error en CV para {nombre}: {e}")

    return pd.DataFrame(resultados)