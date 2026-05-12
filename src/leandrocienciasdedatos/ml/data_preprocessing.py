"""
Funciones de preprocesamiento de datos para Machine Learning.
Prepara el dataset_final_integrado para modelos supervisados y no supervisados.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Columnas definidas ────────────────────────────────────────────────────────
FEATURES_CLASIFICACION = [
    "total_dias_ausencia",
    "total_horas_capacitacion",
    "antiguedad_anos",
    "departamento_encoded",
    "cargo_encoded",
    "tipo_contrato_encoded",
    "jornada_encoded",
]

FEATURES_REGRESION = [
    "avg_tecnicas",
    "avg_blandas",
    "total_dias_ausencia",
    "total_horas_capacitacion",
    "antiguedad_anos",
    "departamento_encoded",
    "cargo_encoded",
    "tipo_contrato_encoded",
    "jornada_encoded",
]

TARGET_CLASIFICACION = "alto_desempeno"
TARGET_REGRESION = "avg_desempeno"


def optimizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce uso de memoria convirtiendo tipos de datos.
    float64 → float32, int64 → int32 donde sea posible.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    logger.info("Tipos de datos optimizados para ahorro de RAM.")
    return df


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset_final_integrado para Machine Learning.
    Usa KNNImputer para preservar todas las filas sin perder información.
    Crea el target de clasificación y optimiza tipos de datos.
    """
    from sklearn.impute import KNNImputer

    df = df.copy()

    # Seleccionar columnas numéricas para imputar
    cols_numericas = FEATURES_CLASIFICACION + FEATURES_REGRESION + [
        "score_global", "avg_desempeno"
    ]
    cols_numericas = list(set(cols_numericas))
    cols_existentes = [c for c in cols_numericas if c in df.columns]

    # Aplicar KNNImputer (k=5 vecinos más cercanos)
    imputer = KNNImputer(n_neighbors=5)
    df[cols_existentes] = imputer.fit_transform(df[cols_existentes])

    # Crear target de clasificación binario
    mediana = df["score_global"].median()
    df[TARGET_CLASIFICACION] = (df["score_global"] > mediana).astype(int)

    # Optimizar memoria
    df = optimizar_tipos(df)

    logger.info(
        f"Dataset preparado con KNNImputer: {df.shape[0]} filas | "
        f"Clase 1 (alto desempeño): {df[TARGET_CLASIFICACION].sum()} | "
        f"Clase 0: {(df[TARGET_CLASIFICACION] == 0).sum()}"
    )
    return df


def split_clasificacion(df: pd.DataFrame, test_size: float = 0.2):
    """
    Divide el dataset para clasificación.
    Retorna: X_train, X_test, y_train, y_test
    """
    X = df[FEATURES_CLASIFICACION]
    y = df[TARGET_CLASIFICACION]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(
        f"Split clasificación → Train: {len(X_train)} | Test: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


def split_regresion(df: pd.DataFrame, test_size: float = 0.2):
    """
    Divide el dataset para regresión.
    Retorna: X_train, X_test, y_train, y_test
    """
    X = df[FEATURES_REGRESION]
    y = df[TARGET_REGRESION]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logger.info(
        f"Split regresión → Train: {len(X_train)} | Test: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


def preparar_para_unsupervised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara features para PCA y K-Means.
    Retorna solo columnas numéricas normalizadas.
    """
    cols = [
        "total_dias_ausencia_norm",
        "total_horas_capacitacion_norm",
        "antiguedad_anos_norm",
        "avg_desempeno",
        "avg_tecnicas",
        "avg_blandas",
    ]
    cols_existentes = [c for c in cols if c in df.columns]
    df_unsup = df[cols_existentes].dropna()
    df_unsup = optimizar_tipos(df_unsup)
    logger.info(f"Dataset no supervisado: {df_unsup.shape}")
    return df_unsup