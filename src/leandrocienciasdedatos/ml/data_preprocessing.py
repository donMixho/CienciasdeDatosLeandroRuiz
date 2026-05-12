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
    - Rellena con 0 ausencias y capacitaciones faltantes (lógica de negocio)
    - Usa KNNImputer para métricas de desempeño
    - Crea el target de clasificación
    """
    from sklearn.impute import KNNImputer

    df = df.copy()

    # Paso 1: Relleno lógico de negocio (0 = sin registros)
    cols_cero = [
        "total_dias_ausencia",
        "total_horas_capacitacion",
        "total_dias_ausencia_norm",
        "total_horas_capacitacion_norm",
    ]
    for col in cols_cero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Paso 2: KNNImputer para métricas de desempeño
    cols_knn = [
        "avg_desempeno",
        "avg_tecnicas",
        "avg_blandas",
        "score_global",
        "antiguedad_anos",
        "antiguedad_anos_norm",
        "departamento_encoded",
        "cargo_encoded",
        "tipo_contrato_encoded",
        "jornada_encoded",
    ]
    cols_knn_existentes = [c for c in cols_knn if c in df.columns]

    imputer = KNNImputer(n_neighbors=5)
    df[cols_knn_existentes] = imputer.fit_transform(
        df[cols_knn_existentes]
    )

    # Paso 3: Crear target de clasificación
    mediana = df["score_global"].median()
    df[TARGET_CLASIFICACION] = (
        df["score_global"] > mediana
    ).astype(int)

    # Paso 4: Optimizar memoria
    df = optimizar_tipos(df)

    logger.info(
        f"Dataset preparado: {df.shape[0]} filas | "
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