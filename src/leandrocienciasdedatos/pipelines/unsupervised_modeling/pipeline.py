"""
Pipeline de aprendizaje no supervisado (EV2).
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import aplicar_pca, aplicar_kmeans


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aplicar_pca,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="pca_resultado",
                name="aplicar_pca_node",
            ),
            node(
                func=aplicar_kmeans,
                inputs=["dataset_ml_preparado", "params:ml"],
                outputs="kmeans_resultado",
                name="aplicar_kmeans_node",
            ),
        ]
    )