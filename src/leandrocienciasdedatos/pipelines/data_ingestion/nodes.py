"""
Nodo de ingesta y exploración inicial de datos (AD 1.1).
Genera un reporte diagnóstico de los 4 datasets.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def explorar_dataset(df: pd.DataFrame, nombre: str) -> dict:
    """
    Genera un diagnóstico completo de un DataFrame.
    Retorna un diccionario con métricas clave del dataset.

    Args:
        df: DataFrame a explorar.
        nombre: Nombre descriptivo del dataset.

    Returns:
        Diccionario con el reporte de diagnóstico.
    """
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)

    reporte = {
        "dataset": nombre,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "duplicados": int(df.duplicated().sum()),
        "nulos_por_columna": nulos[nulos > 0].to_dict(),
        "porcentaje_nulos": porcentaje_nulos[porcentaje_nulos > 0].to_dict(),
        "tipos_de_datos": df.dtypes.astype(str).to_dict(),
    }

    logger.info("=" * 50)
    logger.info(f"DIAGNÓSTICO: {nombre}")
    logger.info(f"Shape: {df.shape[0]} filas x {df.shape[1]} columnas")
    logger.info(f"Duplicados: {reporte['duplicados']}")
    logger.info(f"Columnas con nulos: {list(reporte['nulos_por_columna'].keys())}")
    logger.info("=" * 50)

    return reporte


def generar_reporte_diagnostico(
    empleados: pd.DataFrame,
    evaluaciones: pd.DataFrame,
    capacitaciones: pd.DataFrame,
    ausencias: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera un reporte consolidado de diagnóstico de los 4 datasets.

    Args:
        empleados: Dataset de empleados.
        evaluaciones: Dataset de evaluaciones.
        capacitaciones: Dataset de capacitaciones.
        ausencias: Dataset de ausencias.

    Returns:
        DataFrame con el resumen diagnóstico de todos los datasets.
    """
    reportes = []

    for df, nombre in [
        (empleados, "empleados"),
        (evaluaciones, "evaluaciones"),
        (capacitaciones, "capacitaciones"),
        (ausencias, "ausencias"),
    ]:
        r = explorar_dataset(df, nombre)
        reportes.append({
            "dataset": r["dataset"],
            "filas": r["filas"],
            "columnas": r["columnas"],
            "duplicados": r["duplicados"],
            "columnas_con_nulos": len(r["nulos_por_columna"]),
        })

    reporte_df = pd.DataFrame(reportes)
    logger.info("Reporte de diagnóstico generado exitosamente.")
    return reporte_df