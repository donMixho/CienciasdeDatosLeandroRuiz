"""
Optimización de hiperparámetros con GridSearchCV y RandomizedSearchCV.
"""
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)


def get_param_grid_clasificacion() -> dict:
    """Grillas de hiperparámetros para modelos de clasificación."""
    return {
        "RandomForest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [3, 5],
        },
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
    }


def get_param_grid_regresion() -> dict:
    """Grillas de hiperparámetros para modelos de regresión."""
    return {
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
        },
        "RandomForestRegressor": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
    }


def aplicar_grid_search(pipeline, param_grid: dict, X_train: pd.DataFrame,
    y_train: pd.Series, cv: int = 5,
    scoring: str = "f1") -> GridSearchCV:
    """
    Aplica GridSearchCV exhaustivo al pipeline dado.
    Retorna el objeto GridSearchCV entrenado.
    """
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    logger.info(f"GridSearchCV → Mejores params: {gs.best_params_} | "
                f"Mejor score: {round(gs.best_score_, 4)}")
    return gs


def aplicar_randomized_search(pipeline, param_distributions: dict,
    X_train: pd.DataFrame, y_train: pd.Series,
    n_iter: int = 20, cv: int = 5,
    scoring: str = "f1") -> RandomizedSearchCV:
    """
    Aplica RandomizedSearchCV al pipeline dado.
    Más rápido que GridSearch para espacios grandes.
    """
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rs.fit(X_train, y_train)
    logger.info(f"RandomizedSearchCV → Mejores params: {rs.best_params_} | "
                f"Mejor score: {round(rs.best_score_, 4)}")
    return rs


def comparar_resultados_tuning(gs_result, rs_result,
    nombre_modelo: str) -> pd.DataFrame:
    """
    Compara resultados de GridSearch vs RandomizedSearch para un modelo.
    """
    return pd.DataFrame([
        {
            "modelo": nombre_modelo,
            "metodo": "GridSearchCV",
            "mejor_score": round(gs_result.best_score_, 4),
            "mejores_params": str(gs_result.best_params_),
        },
        {
            "modelo": nombre_modelo,
            "metodo": "RandomizedSearchCV",
            "mejor_score": round(rs_result.best_score_, 4),
            "mejores_params": str(rs_result.best_params_),
        },
    ])