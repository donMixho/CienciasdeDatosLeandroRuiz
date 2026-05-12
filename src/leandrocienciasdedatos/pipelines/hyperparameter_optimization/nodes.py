"""
Nodos del pipeline de optimización de hiperparámetros (EV2).
Aplica GridSearchCV y RandomizedSearchCV sobre los mejores modelos.
"""
import logging
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leandrocienciasdedatos.ml.data_preprocessing import (
    split_clasificacion,
    split_regresion,
)
from leandrocienciasdedatos.ml.model_training import (
    get_modelos_clasificacion,
    get_modelos_regresion,
)
from leandrocienciasdedatos.ml.hyperparameter_tuning import (
    get_param_grid_clasificacion,
    get_param_grid_regresion,
    aplicar_grid_search,
    aplicar_randomized_search,
    comparar_resultados_tuning,
)

logger = logging.getLogger(__name__)
MODELS_PATH = "data/06_models"
PLOTS_PATH = "results/plots"
METRICS_PATH = "results/metrics"


def optimizar_clasificacion(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Aplica GridSearchCV y RandomizedSearchCV a los modelos
    de clasificación definidos en param_grids.
    Guarda los mejores modelos optimizados.
    Retorna DataFrame comparativo de resultados.
    """
    cv = parameters.get("cv_folds", 5)
    n_iter = parameters.get("random_search_n_iter", 20)

    X_train, X_test, y_train, y_test = split_clasificacion(dataset_ml)

    modelos = get_modelos_clasificacion()
    param_grids = get_param_grid_clasificacion()

    resultados = []
    os.makedirs(MODELS_PATH, exist_ok=True)

    for nombre, param_grid in param_grids.items():
        if nombre not in modelos:
            continue

        pipeline_base = modelos[nombre]
        logger.info(f"Optimizando: {nombre}")

        # GridSearchCV
        gs = aplicar_grid_search(
            pipeline_base, param_grid, X_train, y_train,
            cv=cv, scoring="f1"
        )

        # RandomizedSearchCV
        rs = aplicar_randomized_search(
            pipeline_base, param_grid, X_train, y_train,
            n_iter=n_iter, cv=cv, scoring="f1"
        )

        # Guardar mejores modelos
        joblib.dump(
            gs.best_estimator_,
            os.path.join(MODELS_PATH, f"{nombre}_clf_grid_best.pkl")
        )
        joblib.dump(
            rs.best_estimator_,
            os.path.join(MODELS_PATH, f"{nombre}_clf_random_best.pkl")
        )

        df_comp = comparar_resultados_tuning(gs, rs, nombre)
        resultados.append(df_comp)

    if not resultados:
        return pd.DataFrame()

    df_final = pd.concat(resultados, ignore_index=True)

    # Gráfico comparativo GridSearch vs RandomizedSearch
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos_nombres = df_final["modelo"].unique()
    x = range(len(modelos_nombres))
    width = 0.35

    scores_grid = df_final[df_final["metodo"] == "GridSearchCV"]["mejor_score"].values
    scores_random = df_final[df_final["metodo"] == "RandomizedSearchCV"]["mejor_score"].values

    ax.bar([i - width/2 for i in x], scores_grid, width,
            label="GridSearchCV", color="steelblue", alpha=0.8)
    ax.bar([i + width/2 for i in x], scores_random, width,
            label="RandomizedSearchCV", color="darkorange", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(modelos_nombres, rotation=15)
    ax.set_ylabel("F1 Score (CV)")
    ax.set_title("Optimización Clasificación: Grid vs Randomized Search")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_PATH, "tuning_comparacion_clasificacion.png"),
        dpi=100
    )
    plt.close()

    os.makedirs(METRICS_PATH, exist_ok=True)
    df_final.to_csv(
        os.path.join(METRICS_PATH, "tuning_clasificacion.csv"),
        index=False, encoding="utf-8"
    )

    logger.info(f"Optimización clasificación completada. {len(param_grids)} modelos.")
    return df_final


def optimizar_regresion(
    dataset_ml: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """
    Aplica GridSearchCV y RandomizedSearchCV a los modelos
    de regresión definidos en param_grids.
    Guarda los mejores modelos optimizados.
    Retorna DataFrame comparativo de resultados.
    """
    cv = parameters.get("cv_folds", 5)
    n_iter = parameters.get("random_search_n_iter", 20)

    X_train, X_test, y_train, y_test = split_regresion(dataset_ml)

    modelos = get_modelos_regresion()
    param_grids = get_param_grid_regresion()

    resultados = []
    os.makedirs(MODELS_PATH, exist_ok=True)

    for nombre, param_grid in param_grids.items():
        if nombre not in modelos:
            continue

        pipeline_base = modelos[nombre]
        logger.info(f"Optimizando: {nombre}")

        gs = aplicar_grid_search(
            pipeline_base, param_grid, X_train, y_train,
            cv=cv, scoring="r2"
        )

        rs = aplicar_randomized_search(
            pipeline_base, param_grid, X_train, y_train,
            n_iter=n_iter, cv=cv, scoring="r2"
        )

        joblib.dump(
            gs.best_estimator_,
            os.path.join(MODELS_PATH, f"{nombre}_reg_grid_best.pkl")
        )
        joblib.dump(
            rs.best_estimator_,
            os.path.join(MODELS_PATH, f"{nombre}_reg_random_best.pkl")
        )

        df_comp = comparar_resultados_tuning(gs, rs, nombre)
        resultados.append(df_comp)

    if not resultados:
        return pd.DataFrame()

    df_final = pd.concat(resultados, ignore_index=True)

    # Gráfico comparativo
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos_nombres = df_final["modelo"].unique()
    x = range(len(modelos_nombres))
    width = 0.35

    scores_grid = df_final[df_final["metodo"] == "GridSearchCV"]["mejor_score"].values
    scores_random = df_final[df_final["metodo"] == "RandomizedSearchCV"]["mejor_score"].values

    ax.bar([i - width/2 for i in x], scores_grid, width,
            label="GridSearchCV", color="seagreen", alpha=0.8)
    ax.bar([i + width/2 for i in x], scores_random, width,
            label="RandomizedSearchCV", color="mediumpurple", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(modelos_nombres, rotation=15)
    ax.set_ylabel("R2 Score (CV)")
    ax.set_title("Optimización Regresión: Grid vs Randomized Search")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_PATH, "tuning_comparacion_regresion.png"),
        dpi=100
    )
    plt.close()

    df_final.to_csv(
        os.path.join(METRICS_PATH, "tuning_regresion.csv"),
        index=False, encoding="utf-8"
    )

    logger.info(f"Optimización regresión completada. {len(param_grids)} modelos.")
    return df_final