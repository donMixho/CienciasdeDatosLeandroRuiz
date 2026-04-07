from kedro.pipeline import Pipeline, node, pipeline
from .nodes import integrate_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=integrate_data,
            inputs=["empleados_clean", "evaluaciones_clean", "capacitaciones_clean", "ausencias_clean"],
            outputs="dataset_final_integrado",
            name="integration_node",
        ),
    ])