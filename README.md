# LeandroCienciasDeDatos — EV2

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

## Descripción

Proyecto de **Machine Learning sobre datos de Recursos Humanos**, desarrollado como parte de la asignatura **SCY1101 - Programación para la Ciencia de Datos** (EV2). Extiende el trabajo de EV1 incorporando modelado supervisado, no supervisado y optimización de hiperparámetros, orquestado con el framework **Kedro**.

Los datos corresponden a una empresa con 4 datasets relacionados:

- `empleados.csv` — Información base de cada trabajador
- `evaluaciones.csv` — Evaluaciones de desempeño por periodo
- `capacitaciones.csv` — Cursos y horas de formación
- `ausencias.csv` — Registro de ausencias laborales

---

## Qué busca aprender el modelo

El proyecto plantea dos problemas de Machine Learning sobre los datos de empleados:

### Clasificación — ¿Este empleado tendrá alto desempeño?

El modelo de clasificación busca predecir si un empleado pertenece al grupo de **alto desempeño** (`alto_desempeno = 1`) o no (`= 0`), en base a características como sus horas de capacitación, días de ausencia, antigüedad, departamento y evaluaciones previas.

El objetivo es que el modelo aprenda a identificar qué perfil de empleado tiende a destacarse, lo que en un contexto real podría usarse para decisiones de desarrollo de talento, retención o promoción.

> El target se construye comparando el `score_global` de cada empleado contra la mediana: quienes están por encima se etiquetan como alto desempeño.

### Regresión — ¿Cuál será el puntaje de desempeño del empleado?

El modelo de regresión busca predecir el valor numérico de `avg_desempeno` (promedio de evaluaciones de desempeño), en lugar de solo clasificarlo en categorías.

El objetivo es que el modelo aprenda la relación entre las características del empleado y su nivel de desempeño medido, permitiendo estimar con mayor precisión cuánto puede rendir un empleado dado su perfil.

---

## Contexto y limitaciones del dataset

El dataset de Recursos Humanos utilizado en este proyecto es de **tamaño reducido**, lo que impone limitaciones naturales sobre la capacidad de aprendizaje de los modelos. Con pocos registros, los algoritmos tienen menos ejemplos para generalizar patrones, lo que se refleja principalmente en las métricas de regresión (R² = 0.2795).

### Estrategias para preservar los datos

Dado el volumen limitado, se priorizó no perder registros. En lugar de eliminar filas con valores faltantes, se aplicaron las siguientes técnicas:

- **KNNImputer (k=5)**: imputa los valores faltantes en métricas de desempeño (`avg_desempeno`, `avg_tecnicas`, `avg_blandas`, `score_global`) utilizando los 5 vecinos más cercanos. Esto permite conservar cada empleado en el dataset en lugar de descartarlo por tener datos incompletos.
- **Imputación por lógica de negocio**: ausencias y horas de capacitación faltantes se rellenan con 0, ya que la ausencia de registro indica que el empleado no tuvo ausencias ni capacitaciones.

### Estrategias para mejorar el aprendizaje con pocos datos

- **`class_weight='balanced'`** en clasificación: compensa el desbalance entre clases cuando hay pocas muestras de una categoría.
- **Métodos de ensemble** (RandomForest, GradientBoosting, XGBoost, StackingEnsemble): combinan múltiples modelos para reducir varianza y sobreajuste, siendo más robustos con datasets pequeños.
- **Validación cruzada 5-fold**: evalúa cada modelo en 5 particiones distintas del dataset, obteniendo métricas más confiables que una única división train/test.
- **Split estratificado** (`stratify=y`): garantiza que ambas clases estén proporcionalmente representadas en train y test.
- **Regularización** (Ridge, Lasso): penaliza la complejidad del modelo para evitar sobreajuste.
- **Feature engineering**: variables `ratio_cap_antiguedad` y `tasa_ausencia` aportan relaciones informativas adicionales sin requerir más datos.

---

## Resultados principales

| Tarea | Mejor modelo | Métrica |
|---|---|---|
| Clasificación (rotación) | XGBoost | F1 = 0.8148 |
| Regresión (score global) | XGBoostRegressor | R² = 0.2795 |
| Clustering | K-Means (k=3) | PCA 2 componentes |

---

## Estructura del Proyecto

```
CienciasdeDatosLeandroRuiz/
├── conf/
│   └── base/
│       ├── catalog.yml          → Definición de todos los datasets
│       └── parameters.yml       → Parámetros configurables
├── data/
│   ├── 01_raw/                  → CSVs originales (no modificar)
│   ├── 02_intermediate/         → Datos limpios por tabla
│   ├── 03_primary/              → Dataset integrado
│   ├── 04_feature/              → Dataset preparado para ML
│   ├── 07_model_output/         → Métricas, resultados CV y tuning
│   └── 08_reporting/            → Reportes de diagnóstico y validación
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb     → EDA inicial
│   ├── 02_supervised_modeling.ipynb      → Entrenamiento de modelos
│   ├── 03_model_evaluation.ipynb         → Evaluación y comparación
│   ├── 04_hyperparameter_optimization.ipynb → GridSearchCV / RandomizedSearchCV
│   └── 05_final_analysis.ipynb           → Análisis final integrado
├── src/leandrocienciasdedatos/
│   ├── ml/
│   │   ├── data_preprocessing.py         → Preparación del dataset ML
│   │   ├── model_training.py             → Definición y entrenamiento de modelos
│   │   ├── model_evaluation.py           → Métricas y evaluación
│   │   └── hyperparameter_tuning.py      → Búsqueda de hiperparámetros
│   └── pipelines/
│       ├── data_ingestion/               → Pipeline 1: Carga y diagnóstico
│       ├── data_cleaning/                → Pipeline 2: Limpieza
│       ├── data_transformation/          → Pipeline 3: Transformación
│       ├── data_validation/              → Pipeline 4: Validación
│       ├── supervised_modeling/          → Pipeline 5: Modelos supervisados
│       ├── unsupervised_modeling/        → Pipeline 6: PCA + K-Means
│       ├── model_evaluation/             → Pipeline 7: Evaluación con CV
│       └── hyperparameter_optimization/  → Pipeline 8: Tuning
├── results/                              → Outputs finales exportados
├── pyproject.toml
└── README.md
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/donMixho/CienciasdeDatosLeandroRuiz.git
cd CienciasdeDatosLeandroRuiz
```

### 2. Crear y activar entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -e .
```

O con las dependencias opcionales de desarrollo:

```bash
pip install -e ".[dev]"
```

### 4. Agregar los CSV originales

Colocar los 4 archivos en `data/01_raw/`:

- `empleados.csv`
- `evaluaciones.csv`
- `capacitaciones.csv`
- `ausencias.csv`

---

## Ejecución

### Correr todos los pipelines en orden

```bash
kedro run
```

### Correr un pipeline específico

```bash
# Preprocesamiento (EV1)
kedro run --pipeline data_ingestion
kedro run --pipeline data_cleaning
kedro run --pipeline data_transformation
kedro run --pipeline data_validation

# Machine Learning (EV2)
kedro run --pipeline supervised_modeling
kedro run --pipeline unsupervised_modeling
kedro run --pipeline model_evaluation
kedro run --pipeline hyperparameter_optimization
```

### Abrir los notebooks

```bash
kedro jupyter notebook
```

---

## Pipelines

### Pipeline 1 — Data Ingestion

Carga los 4 CSVs y genera un reporte de diagnóstico inicial.

**Output:** `data/08_reporting/reporte_diagnostico.csv`

### Pipeline 2 — Data Cleaning

Limpieza de los 4 datasets:

- Eliminación de duplicados
- Tratamiento de valores nulos
- Corrección de tipos de datos y fechas
- Tratamiento de outliers con método IQR

**Output:** `data/02_intermediate/*_clean.csv`

### Pipeline 3 — Data Transformation

Integración y feature engineering:

- Joins por `id_empleado`
- Features derivadas: `antiguedad_anos`, `score_global`
- Normalización con MinMaxScaler y codificación con LabelEncoder

**Output:** `data/03_primary/dataset_final_integrado.csv`

### Pipeline 4 — Data Validation

Verifica integridad del proceso de limpieza y transformación.

**Output:** `data/08_reporting/reporte_comparacion.csv`, `reporte_validacion_final.csv`

### Pipeline 5 — Supervised Modeling

Entrena 13 modelos (7 clasificación + 6 regresión) con Scikit-learn Pipelines (StandardScaler + modelo). `random_state=42` en todos los modelos.

**Modelos de clasificación** (`target: rotacion`):

| Modelo | Configuración |
|---|---|
| LogisticRegression | `class_weight='balanced'` |
| DecisionTree | `class_weight='balanced'` |
| RandomForest | `n_estimators=100`, `class_weight='balanced'` |
| GradientBoosting | `n_estimators=100`, `learning_rate=0.1` |
| SVM | `probability=True`, `class_weight='balanced'` |
| XGBoost | `n_estimators=100` |
| StackingEnsemble | XGB + GB + SVM → LogisticRegression |

**Modelos de regresión** (`target: score_global`):

| Modelo | Configuración |
|---|---|
| LinearRegression | — |
| Ridge | — |
| Lasso | — |
| DecisionTreeRegressor | — |
| RandomForestRegressor | `n_estimators=100` |
| XGBoostRegressor | `n_estimators=100` |

**Output:** `data/07_model_output/metricas_clasificacion.csv`, `metricas_regresion.csv`

### Pipeline 6 — Unsupervised Modeling

Reducción de dimensionalidad y clustering:

- **PCA**: 2 componentes principales sobre el dataset ML
- **K-Means**: clustering con `n_clusters=3`, `random_state=42`

**Output:** `data/07_model_output/pca_resultado.csv`, `kmeans_resultado.csv`

### Pipeline 7 — Model Evaluation

Validación cruzada (5-fold) de todos los modelos con métricas completas:

- Clasificación: F1, Accuracy, ROC-AUC
- Regresión: R², MAE, MSE, RMSE

**Output:** `data/07_model_output/cv_clasificacion.csv`, `cv_regresion.csv`

### Pipeline 8 — Hyperparameter Optimization

Búsqueda de hiperparámetros sobre los mejores modelos:

- **GridSearchCV**: búsqueda exhaustiva
- **RandomizedSearchCV**: búsqueda aleatoria eficiente

**Output:** `data/07_model_output/tuning_clasificacion.csv`, `tuning_regresion.csv`

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| `kedro` | ~1.3.1 | Orquestación del flujo de datos |
| `pandas` | >=2.3.3 | Manipulación de datos |
| `scikit-learn` | >=1.3.0 | Modelos ML, pipelines, métricas |
| `xgboost` | — | XGBClassifier, XGBRegressor |
| `numpy` | >=1.26.0 | Operaciones numéricas |
| `jupyterlab` | >=3.0 | Notebooks interactivos |

---

## Autor

**Leandro Ruiz**  
Ingeniería Informática — Ciencia de Datos  
SCY1101 — 2025
