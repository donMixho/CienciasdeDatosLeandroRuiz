"""
Definición y entrenamiento de modelos supervisados.
10 modelos: 5 clasificación + 5 regresión usando Pipelines de Scikit-learn.
"""
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)



def get_modelos_clasificacion() -> dict:
    """
    Retorna diccionario con 5 pipelines de clasificación.
    Cada pipeline incluye StandardScaler + modelo.
    random_state=42 en todos los modelos que lo soporten.
    class_weight='balanced' para compensar desbalance de clases.
    """
    modelos = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced"))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(random_state=42, probability=True, class_weight="balanced"))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(random_state=42, n_estimators=100, eval_metric="logloss", verbosity=0))
        ]),

        "StackingEnsemble": Pipeline([
            ("scaler", StandardScaler()),
            ("model", StackingClassifier(
                estimators=[
                    ("xgb", XGBClassifier(
                        random_state=42, n_estimators=100,
                        eval_metric="logloss", verbosity=0
                    )),
                    ("gb", GradientBoostingClassifier(
                        random_state=42, n_estimators=200,
                        learning_rate=0.1, max_depth=4
                    )),
                    ("svm", SVC(
                        random_state=42, probability=True,
                        class_weight="balanced"
                    )),
                ],
                final_estimator=LogisticRegression(random_state=42),
                cv=5
            ))
        ]),

    }
    logger.info(f"Modelos de clasificación definidos: {list(modelos.keys())}")
    return modelos



def get_modelos_regresion() -> dict:
    """
    Retorna diccionario con 5 pipelines de regresión.
    Cada pipeline incluye StandardScaler + modelo.
    """
    modelos = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=42))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=42))
        ]),
        "DecisionTreeRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(random_state=42))
        ]),
        "RandomForestRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=42, n_estimators=100))
        ]),

        "XGBoostRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                random_state=42,
                n_estimators=100,
                eval_metric="rmse",
                verbosity=0
            ))
        ]),


    }
    logger.info(f"Modelos de regresión definidos: {list(modelos.keys())}")
    return modelos


def entrenar_modelos(modelos: dict, X_train: pd.DataFrame,
    y_train: pd.Series) -> dict:
    """
    Entrena todos los modelos del diccionario.
    Retorna diccionario con modelos entrenados.
    """
    modelos_entrenados = {}
    for nombre, pipeline in modelos.items():
        try:
            pipeline.fit(X_train, y_train)
            modelos_entrenados[nombre] = pipeline
            logger.info(f"✅ Entrenado: {nombre}")
        except Exception as e:
            logger.error(f"❌ Error entrenando {nombre}: {e}")
    return modelos_entrenados