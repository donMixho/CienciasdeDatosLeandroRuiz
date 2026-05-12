"""
Pipeline de evaluación de modelos (EV2).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluar_clasificacion_cv, evaluar_regresion_cv


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluar_clasificacion_cv,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="cv_clasificacion",
                name="evaluar_clasificacion_cv_node",
            ),
            node(
                func=evaluar_regresion_cv,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="cv_regresion",
                name="evaluar_regresion_cv_node",
            ),
        ]
    )