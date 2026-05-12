"""
Pipeline de optimización de hiperparámetros (EV2).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import optimizar_clasificacion, optimizar_regresion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=optimizar_clasificacion,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="tuning_clasificacion",
                name="optimizar_clasificacion_node",
            ),
            node(
                func=optimizar_regresion,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="tuning_regresion",
                name="optimizar_regresion_node",
            ),
        ]
    )