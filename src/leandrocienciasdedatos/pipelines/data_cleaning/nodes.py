"""
Nodos de limpieza de datos (AD 1.2).
Trata nulos, duplicados, tipos mixtos, fechas y outliers.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def limpiar_empleados(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Limpia el DataFrame de empleados.

    Args:
        df: DataFrame crudo de empleados.
        parameters: Parámetros desde parameters.yml.

    Returns:
        DataFrame limpio de empleados.
    """
    df = df.copy()
    filas_inicial = len(df)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Eliminar filas sin nombre o rut (datos clave irrecuperables)
    df = df.dropna(subset=["nombre", "rut"])

    # 3. Rellenar nulos en categóricas con valor por defecto
    valor_default = parameters["default_category"]
    for col in ["departamento", "cargo", "tipo_contrato", "jornada"]:
        df[col] = df[col].fillna(valor_default)

    # 4. Corregir tipos: id_empleado a int, salario a float
    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce")
    df["salario"] = pd.to_numeric(df["salario"], errors="coerce")
    df = df.dropna(subset=["id_empleado"])
    df["id_empleado"] = df["id_empleado"].astype(int)

    # 5. Estandarizar fechas con formato mixto a YYYY-MM-DD
    df["fecha_ingreso"] = pd.to_datetime(
        df["fecha_ingreso"], dayfirst=True, errors="coerce"
    )

    # 6. Normalizar strings: strip y title case
    for col in ["nombre", "departamento", "cargo"]:
        df[col] = df[col].str.strip().str.title()

    # 7. Limpiar caracteres especiales problemáticos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x
            )

    logger.info(f"empleados: {filas_inicial} -> {len(df)} filas tras limpieza")
    return df


def limpiar_evaluaciones(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Limpia el DataFrame de evaluaciones.

    Args:
        df: DataFrame crudo de evaluaciones.
        parameters: Parámetros desde parameters.yml.

    Returns:
        DataFrame limpio de evaluaciones.
    """
    df = df.copy()
    filas_inicial = len(df)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Eliminar filas sin id_empleado (clave relacional)
    df = df.dropna(subset=["id_empleado"])

    # 3. Rellenar nulos numéricos con la mediana
    for col in ["puntaje_desempeno", "competencias_tecnicas", "competencias_blandas"]:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)

    # 4. Tratar outliers en puntaje_desempeno con IQR
    threshold = parameters["outlier_threshold"]
    Q1 = df["puntaje_desempeno"].quantile(0.25)
    Q3 = df["puntaje_desempeno"].quantile(0.75)
    IQR = Q3 - Q1
    df["puntaje_desempeno"] = df["puntaje_desempeno"].clip(
        Q1 - threshold * IQR,
        Q3 + threshold * IQR,
    )

    # 5. Corregir tipos
    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").astype(int)
    df["id_evaluacion"] = pd.to_numeric(df["id_evaluacion"], errors="coerce")

    # 6. Rellenar nulos en texto
    df["periodo"] = df["periodo"].fillna("Sin Periodo")
    df["evaluador"] = df["evaluador"].fillna("Sin Evaluador")

    # Limpiar caracteres especiales problemáticos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x
            )

    logger.info(f"evaluaciones: {filas_inicial} -> {len(df)} filas tras limpieza")
    return df


def limpiar_capacitaciones(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Limpia el DataFrame de capacitaciones.

    Args:
        df: DataFrame crudo de capacitaciones.
        parameters: Parámetros desde parameters.yml.

    Returns:
        DataFrame limpio de capacitaciones.
    """
    df = df.copy()
    filas_inicial = len(df)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Eliminar filas sin id_empleado
    df = df.dropna(subset=["id_empleado"])

    # 3. Rellenar nulos en texto
    df["nombre_curso"] = df["nombre_curso"].fillna("Sin Nombre")
    df["estado"] = df["estado"].fillna("Sin Estado")

    # 4. Rellenar nulos numéricos con mediana
    df["horas"] = df["horas"].fillna(df["horas"].median())
    df["nota_final"] = df["nota_final"].fillna(df["nota_final"].median())

    # 5. Estandarizar fechas con formato mixto
    df["fecha_inicio"] = pd.to_datetime(
        df["fecha_inicio"], dayfirst=True, errors="coerce"
    )
    df["fecha_fin"] = pd.to_datetime(
        df["fecha_fin"], dayfirst=True, errors="coerce"
    )

    # 6. Corregir tipos
    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").astype(int)

    # 7. Limpiar caracteres especiales problemáticos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x
            )

    logger.info(f"capacitaciones: {filas_inicial} -> {len(df)} filas tras limpieza")
    return df


def limpiar_ausencias(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Limpia el DataFrame de ausencias.

    Args:
        df: DataFrame crudo de ausencias.
        parameters: Parámetros desde parameters.yml.

    Returns:
        DataFrame limpio de ausencias.
    """
    df = df.copy()
    filas_inicial = len(df)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Eliminar filas sin id_empleado
    df = df.dropna(subset=["id_empleado"])

    # 3. Estandarizar fechas (incluye timestamps con hora)
    df["fecha_inicio"] = pd.to_datetime(
        df["fecha_inicio"], dayfirst=True, errors="coerce"
    )
    df["fecha_fin"] = pd.to_datetime(
        df["fecha_fin"], dayfirst=True, errors="coerce"
    )

    # 4. Rellenar nulos en texto
    valor_binario = parameters["default_binary"]
    df["tipo_ausencia"] = df["tipo_ausencia"].fillna("Sin Tipo")
    df["justificada"] = df["justificada"].fillna(valor_binario)

    # 5. Rellenar nulos en dias con mediana
    df["dias"] = df["dias"].fillna(df["dias"].median())

    # 6. Corregir tipos
    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").astype(int)

    # 7. Limpiar caracteres especiales problemáticos (para Windows/CP1252)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x
            )

    logger.info(f"ausencias: {filas_inicial} -> {len(df)} filas tras limpieza")
    return df