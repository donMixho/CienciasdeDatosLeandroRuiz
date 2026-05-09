"""
Pipeline de validación de datos (AD 1.4).
Verifica integridad, esquemas y comparación antes/después.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import validar_dataset_final, validar_integridad


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validar_integridad,
                inputs=[
                    "empleados",
                    "empleados_clean",
                    "evaluaciones",
                    "evaluaciones_clean",
                    "capacitaciones",
                    "capacitaciones_clean",
                    "ausencias",
                    "ausencias_clean",
                ],
                outputs="reporte_comparacion",
                name="validar_integridad_node",
            ),
            node(
                func=validar_dataset_final,
                inputs="dataset_final_integrado",
                outputs="reporte_validacion_final",
                name="validar_dataset_final_node",
            ),
        ]
    )