"""
Pipeline de limpieza de datos (AD 1.2).
Limpia los 4 datasets y los guarda en 02_intermediate.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    limpiar_ausencias,
    limpiar_capacitaciones,
    limpiar_empleados,
    limpiar_evaluaciones,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=limpiar_empleados,
                inputs=["empleados", "params:cleaning"],
                outputs="empleados_clean",
                name="limpiar_empleados_node",
            ),
            node(
                func=limpiar_evaluaciones,
                inputs=["evaluaciones", "params:cleaning"],
                outputs="evaluaciones_clean",
                name="limpiar_evaluaciones_node",
            ),
            node(
                func=limpiar_capacitaciones,
                inputs=["capacitaciones", "params:cleaning"],
                outputs="capacitaciones_clean",
                name="limpiar_capacitaciones_node",
            ),
            node(
                func=limpiar_ausencias,
                inputs=["ausencias", "params:cleaning"],
                outputs="ausencias_clean",
                name="limpiar_ausencias_node",
            ),
        ]
    )