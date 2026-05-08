"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from leandrocienciasdedatos.pipelines.data_cleaning.pipeline import (
    create_pipeline as create_data_cleaning_pipeline,
)
from leandrocienciasdedatos.pipelines.data_ingestion.pipeline import (
    create_pipeline as create_data_ingestion_pipeline,
)
from leandrocienciasdedatos.pipelines.data_transformation.pipeline import (
    create_pipeline as create_data_transformation_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "data_ingestion": create_data_ingestion_pipeline(),
        "data_cleaning": create_data_cleaning_pipeline(),
        "data_transformation": create_data_transformation_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
