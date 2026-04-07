from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs="empleados",
            outputs="empleados_clean",
            name="clean_empleados_node",
        ),
    ])