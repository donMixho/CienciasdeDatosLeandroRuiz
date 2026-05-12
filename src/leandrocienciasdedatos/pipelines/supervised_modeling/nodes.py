"""
Nodos del pipeline de modelado supervisado (EV2).
Entrena 5 modelos de clasificación y 5 de regresión.
"""
import logging
import os
import joblib
import pandas as pd

from leandrocienciasdedatos.ml.data_preprocessing import (
    preparar_dataset,
    split_clasificacion,
    split_regresion,
)
from leandrocienciasdedatos.ml.model_training import (
    get_modelos_clasificacion,
    get_modelos_regresion,
    entrenar_modelos,
)
from leandrocienciasdedatos.ml.model_evaluation import (
    evaluar_clasificacion,
    evaluar_regresion,
)

logger = logging.getLogger(__name__)
MODELS_PATH = "data/06_models"


def preparar_datos_ml(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset_final_integrado para Machine Learning.
    Crea el target de clasificación y optimiza tipos de datos.
    """
    return preparar_dataset(dataset)


def entrenar_modelos_clasificacion(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Entrena 5 modelos de clasificación con Pipelines de Scikit-learn.
    Guarda los modelos en data/06_models/.
    Retorna DataFrame con métricas en test set.
    """
    test_size = parameters.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = split_clasificacion(dataset_ml, test_size)

    modelos = get_modelos_clasificacion()
    modelos_entrenados = entrenar_modelos(modelos, X_train, y_train)

    # Persistir modelos en disco
    os.makedirs(MODELS_PATH, exist_ok=True)
    for nombre, modelo in modelos_entrenados.items():
        ruta = os.path.join(MODELS_PATH, f"{nombre}_clf.pkl")
        joblib.dump(modelo, ruta)
        logger.info(f"Modelo guardado: {ruta}")

    df_metricas = evaluar_clasificacion(modelos_entrenados, X_test, y_test)
    logger.info(f"Mejor F1: {df_metricas['f1_score'].max()}")
    return df_metricas


def entrenar_modelos_regresion(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Entrena 5 modelos de regresión con Pipelines de Scikit-learn.
    Guarda los modelos en data/06_models/.
    Retorna DataFrame con métricas en test set.
    """
    test_size = parameters.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = split_regresion(dataset_ml, test_size)

    modelos = get_modelos_regresion()
    modelos_entrenados = entrenar_modelos(modelos, X_train, y_train)

    os.makedirs(MODELS_PATH, exist_ok=True)
    for nombre, modelo in modelos_entrenados.items():
        ruta = os.path.join(MODELS_PATH, f"{nombre}_reg.pkl")
        joblib.dump(modelo, ruta)
        logger.info(f"Modelo guardado: {ruta}")

    df_metricas = evaluar_regresion(modelos_entrenados, X_test, y_test)
    logger.info(f"Mejor R2: {df_metricas['r2'].max()}")
    return df_metricas