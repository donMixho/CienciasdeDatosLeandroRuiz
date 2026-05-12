"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

# Pipelines EV1
from leandrocienciasdedatos.pipelines.data_cleaning.pipeline import (
    create_pipeline as create_data_cleaning_pipeline,
)
from leandrocienciasdedatos.pipelines.data_ingestion.pipeline import (
    create_pipeline as create_data_ingestion_pipeline,
)
from leandrocienciasdedatos.pipelines.data_transformation.pipeline import (
    create_pipeline as create_data_transformation_pipeline,
)
from leandrocienciasdedatos.pipelines.data_validation.pipeline import (
    create_pipeline as create_data_validation_pipeline,
)

# Pipelines EV2
from leandrocienciasdedatos.pipelines.supervised_modeling.pipeline import (
    create_pipeline as create_supervised_modeling_pipeline,
)
from leandrocienciasdedatos.pipelines.unsupervised_modeling.pipeline import (
    create_pipeline as create_unsupervised_modeling_pipeline,
)
from leandrocienciasdedatos.pipelines.model_evaluation.pipeline import (
    create_pipeline as create_model_evaluation_pipeline,
)
from leandrocienciasdedatos.pipelines.hyperparameter_optimization.pipeline import (
    create_pipeline as create_hyperparameter_optimization_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""

    pipelines = {
        # EV1
        "data_ingestion": create_data_ingestion_pipeline(),
        "data_cleaning": create_data_cleaning_pipeline(),
        "data_transformation": create_data_transformation_pipeline(),
        "data_validation": create_data_validation_pipeline(),
        # EV2
        "supervised_modeling": create_supervised_modeling_pipeline(),
        "unsupervised_modeling": create_unsupervised_modeling_pipeline(),
        "model_evaluation": create_model_evaluation_pipeline(),
        "hyperparameter_optimization": create_hyperparameter_optimization_pipeline(),
    }

    pipelines["__default__"] = sum(pipelines.values())
    return pipelines