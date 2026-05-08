"""
Pipeline de transformación e integración de datos (AD 1.3).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import integrar_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=integrar_datasets,
                inputs=[
                    "empleados_clean",
                    "evaluaciones_clean",
                    "capacitaciones_clean",
                    "ausencias_clean",
                    "params:transform",
                ],
                outputs="dataset_final_integrado",
                name="integrar_datasets_node",
            ),
        ]
    )