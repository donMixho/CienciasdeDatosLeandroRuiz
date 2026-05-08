"""
Pipeline de ingesta y exploración inicial (AD 1.1).
Carga los 4 CSV y genera un reporte diagnóstico.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generar_reporte_diagnostico


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generar_reporte_diagnostico,
                inputs=[
                    "empleados",
                    "evaluaciones",
                    "capacitaciones",
                    "ausencias",
                ],
                outputs="reporte_diagnostico",
                name="generar_reporte_diagnostico_node",
            ),
        ]
    )   