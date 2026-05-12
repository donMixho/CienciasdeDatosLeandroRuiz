"""
Pipeline de modelado supervisado (EV2).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preparar_datos_ml,
    entrenar_modelos_clasificacion,
    entrenar_modelos_regresion,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preparar_datos_ml,
                inputs="dataset_final_integrado",
                outputs="dataset_ml_preparado",
                name="preparar_datos_ml_node",
            ),
            node(
                func=entrenar_modelos_clasificacion,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="metricas_clasificacion",
                name="entrenar_modelos_clasificacion_node",
            ),
            node(
                func=entrenar_modelos_regresion,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="metricas_regresion",
                name="entrenar_modelos_regresion_node",
            ),
        ]
    )