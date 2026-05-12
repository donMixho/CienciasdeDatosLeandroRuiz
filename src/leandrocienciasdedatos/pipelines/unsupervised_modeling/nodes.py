"""
Nodos del pipeline de aprendizaje no supervisado (EV2).
Implementa PCA y K-Means con visualizaciones.
"""
import logging
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Sin interfaz gráfica para Kedro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from leandrocienciasdedatos.ml.data_preprocessing import preparar_para_unsupervised

logger = logging.getLogger(__name__)
MODELS_PATH = "data/06_models"
PLOTS_PATH = "results/plots"


def aplicar_pca(dataset_ml: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Aplica PCA al dataset para reducción de dimensionalidad.
    Guarda el modelo PCA y un gráfico de varianza explicada.
    Retorna DataFrame con las 2 primeras componentes principales.
    """
    n_components = parameters.get("pca_n_components", 2)

    df_unsup = preparar_para_unsupervised(dataset_ml)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_unsup)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Guardar modelo
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(pca, os.path.join(MODELS_PATH, "pca_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler_unsupervised.pkl"))

    # Gráfico varianza explicada
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    varianza = pca.explained_variance_ratio_ * 100
    ax.bar(range(1, len(varianza) + 1), varianza, color="steelblue")
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Varianza Explicada (%)")
    ax.set_title("PCA — Varianza Explicada por Componente")
    for i, v in enumerate(varianza):
        ax.text(i + 1, v + 0.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "pca_varianza_explicada.png"), dpi=100)
    plt.close()

    # Retornar componentes como DataFrame
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)

    varianza_acumulada = float(np.sum(pca.explained_variance_ratio_) * 100)
    logger.info(
        f"PCA aplicado → {n_components} componentes | "
        f"Varianza explicada acumulada: {varianza_acumulada:.1f}%"
    )
    return df_pca


def aplicar_kmeans(dataset_ml: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Aplica K-Means clustering al dataset.
    Determina el número óptimo de clusters con el método del codo.
    Guarda gráficos del codo y de los clusters.
    Retorna DataFrame con asignación de clusters.
    """
    n_clusters = parameters.get("kmeans_n_clusters", 3)
    max_k = parameters.get("kmeans_max_k", 8)

    df_unsup = preparar_para_unsupervised(dataset_ml)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_unsup)

    # Método del codo para encontrar k óptimo
    inercias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inercias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Gráfico del codo
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inercias, "bo-", linewidth=2)
    axes[0].set_xlabel("Número de Clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Método del Codo")
    axes[0].axvline(x=n_clusters, color="red", linestyle="--",
                    label=f"k={n_clusters} seleccionado")
    axes[0].legend()

    axes[1].plot(list(k_range), silhouettes, "go-", linewidth=2)
    axes[1].set_xlabel("Número de Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score por k")
    axes[1].axvline(x=n_clusters, color="red", linestyle="--",
                    label=f"k={n_clusters} seleccionado")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "kmeans_codo_silhouette.png"), dpi=100)
    plt.close()

    # Entrenar K-Means con k seleccionado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Guardar modelo
    joblib.dump(kmeans, os.path.join(MODELS_PATH, "kmeans_model.pkl"))

    # Visualización clusters con PCA 2D
    pca_viz = PCA(n_components=2, random_state=42)
    X_2d = pca_viz.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                        cmap="viridis", alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"K-Means Clustering (k={n_clusters}) — Proyección PCA 2D")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "kmeans_clusters_pca.png"), dpi=100)
    plt.close()

    # Construir DataFrame resultado
    df_resultado = df_unsup.copy()
    df_resultado["cluster"] = labels

    silhouette_final = silhouette_score(X_scaled, labels)
    logger.info(
        f"K-Means k={n_clusters} | "
        f"Silhouette Score: {silhouette_final:.4f}"
    )
    return df_resultado