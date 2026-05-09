"""
Nodos de validación de datos (AD 1.4).
Verifica integridad, esquemas y comparación antes/después.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def validar_integridad(
    empleados_raw: pd.DataFrame,
    empleados_clean: pd.DataFrame,
    evaluaciones_raw: pd.DataFrame,
    evaluaciones_clean: pd.DataFrame,
    capacitaciones_raw: pd.DataFrame,
    capacitaciones_clean: pd.DataFrame,
    ausencias_raw: pd.DataFrame,
    ausencias_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara los datasets antes y después de la limpieza.
    Verifica que no haya pérdida inesperada de datos.

    Returns:
        DataFrame con comparación antes/después por dataset.
    """
    comparaciones = []

    pares = [
        ("empleados", empleados_raw, empleados_clean),
        ("evaluaciones", evaluaciones_raw, evaluaciones_clean),
        ("capacitaciones", capacitaciones_raw, capacitaciones_clean),
        ("ausencias", ausencias_raw, ausencias_clean),
    ]

    for nombre, raw, clean in pares:
        nulos_raw = raw.isnull().sum().sum()
        nulos_clean = clean.isnull().sum().sum()
        duplicados_raw = raw.duplicated().sum()
        duplicados_clean = clean.duplicated().sum()

        comparaciones.append({
            "dataset": nombre,
            "filas_antes": len(raw),
            "filas_despues": len(clean),
            "filas_eliminadas": len(raw) - len(clean),
            "nulos_antes": int(nulos_raw),
            "nulos_despues": int(nulos_clean),
            "duplicados_antes": int(duplicados_raw),
            "duplicados_despues": int(duplicados_clean),
            "columnas_antes": raw.shape[1],
            "columnas_despues": clean.shape[1],
        })

        logger.info(
            f"{nombre}: {len(raw)} -> {len(clean)} filas | "
            f"Nulos: {nulos_raw} -> {nulos_clean} | "
            f"Duplicados: {duplicados_raw} -> {duplicados_clean}"
        )

    return pd.DataFrame(comparaciones)


def validar_dataset_final(dataset_final: pd.DataFrame) -> pd.DataFrame:
    """
    Valida el esquema y la integridad del dataset final integrado.
    Verifica columnas esperadas y ausencia de nulos en columnas clave.

    Returns:
        DataFrame con el reporte de validación final.
    """
    columnas_esperadas = [
        "id_empleado", "nombre", "rut", "departamento", "cargo",
        "avg_desempeno", "avg_tecnicas", "avg_blandas",
        "total_dias_ausencia", "total_horas_capacitacion",
        "antiguedad_anos", "score_global",
    ]

    resultados = []

    # Verificar columnas esperadas
    for col in columnas_esperadas:
        existe = col in dataset_final.columns
        nulos = int(dataset_final[col].isnull().sum()) if existe else -1
        resultados.append({
            "columna": col,
            "existe": existe,
            "nulos": nulos,
            "tipo": str(dataset_final[col].dtype) if existe else "N/A",
        })
        if not existe:
            logger.warning(f"Columna esperada no encontrada: {col}")

    # Validar columnas clave sin nulos
    cols_clave = ["id_empleado", "nombre", "rut"]
    nulos_clave = dataset_final[cols_clave].isnull().sum().sum()

    if nulos_clave == 0:
        logger.info("✅ Validación exitosa: columnas clave sin nulos")
    else:
        logger.warning(f"⚠️ Nulos en columnas clave: {nulos_clave}")

    logger.info(
        f"Dataset final validado: {dataset_final.shape[0]} filas "
        f"x {dataset_final.shape[1]} columnas"
    )

    return pd.DataFrame(resultados)