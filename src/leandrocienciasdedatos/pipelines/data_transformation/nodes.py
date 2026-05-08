"""
Nodos de transformación e integración de datos (AD 1.3).
Realiza joins, aggregations, features derivadas y encodings.
"""
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


def integrar_datasets(
    empleados: pd.DataFrame,
    evaluaciones: pd.DataFrame,
    capacitaciones: pd.DataFrame,
    ausencias: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Integra los 4 datasets limpios en un dataset final consolidado.
    Aplica joins, aggregations, features derivadas, normalización
    y codificación de variables categóricas.

    Args:
        empleados: Dataset limpio de empleados.
        evaluaciones: Dataset limpio de evaluaciones.
        capacitaciones: Dataset limpio de capacitaciones.
        ausencias: Dataset limpio de ausencias.
        parameters: Parámetros desde parameters.yml.

    Returns:
        DataFrame final integrado y transformado.
    """
    pk = parameters["primary_key"]
    metricas = parameters["evaluation_metrics"]

    # ── 1. Aggregations con groupby ───────────────────────────────────────
    # Promedio de métricas de evaluación por empleado
    agg_eval = {m: "mean" for m in metricas}
    df_eval = (
        evaluaciones.groupby(pk)
        .agg(agg_eval)
        .reset_index()
        .rename(columns={
            "puntaje_desempeno": "avg_desempeno",
            "competencias_tecnicas": "avg_tecnicas",
            "competencias_blandas": "avg_blandas",
        })
    )

    # Total de días de ausencia por empleado
    df_aus = (
        ausencias.groupby(pk)
        .agg(total_dias_ausencia=("dias", "sum"))
        .reset_index()
    )

    # Total de horas de capacitación por empleado
    df_cap = (
        capacitaciones.groupby(pk)
        .agg(total_horas_capacitacion=("horas", "sum"))
        .reset_index()
    )

    # ── 2. Joins con empleados como tabla base ────────────────────────────
    df = empleados.merge(df_eval, on=pk, how="left")
    df = df.merge(df_aus, on=pk, how="left")
    df = df.merge(df_cap, on=pk, how="left")

    # Rellenar con 0 a quienes no tengan ausencias o capacitaciones
    df["total_dias_ausencia"] = df["total_dias_ausencia"].fillna(0)
    df["total_horas_capacitacion"] = df["total_horas_capacitacion"].fillna(0)

    logger.info(f"Dataset integrado: {df.shape[0]} filas x {df.shape[1]} columnas")

    # ── 3. Features derivadas ─────────────────────────────────────────────
    # Años de antigüedad en la empresa
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["antiguedad_anos"] = (
        (pd.Timestamp.now() - df["fecha_ingreso"]).dt.days / 365
    ).round(1)

    # Score de desempeño global ponderado
    df["score_global"] = (
        df["avg_desempeno"].fillna(0) * 0.5
        + df["avg_tecnicas"].fillna(0) * 0.3
        + df["avg_blandas"].fillna(0) * 0.2
    ).round(3)

    logger.info("Features derivadas creadas: antiguedad_anos, score_global")

    # ── 4. Normalización de columnas numéricas (MinMaxScaler) ─────────────
    cols_normalizar = [
        "total_dias_ausencia",
        "total_horas_capacitacion",
        "antiguedad_anos",
    ]
    scaler = MinMaxScaler()
    cols_existentes = [c for c in cols_normalizar if c in df.columns]
    df[[f"{c}_norm" for c in cols_existentes]] = scaler.fit_transform(
        df[cols_existentes].fillna(0)
    )

    logger.info(f"Columnas normalizadas: {cols_existentes}")

    # ── 5. Codificación de variables categóricas (Label Encoding) ─────────
    cols_categoricas = ["departamento", "cargo", "tipo_contrato", "jornada"]
    le = LabelEncoder()
    for col in cols_categoricas:
        if col in df.columns:
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

    logger.info(f"Variables codificadas: {cols_categoricas}")

    logger.info(f"Dataset final: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df