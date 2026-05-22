# LeandroCienciasDeDatos — EV2

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

## Descripción

Este es mi proyecto de **Machine Learning aplicado a datos de Recursos Humanos**, desarrollado para la asignatura **SCY1101 - Programación para la Ciencia de Datos** (EV2). Es la continuación de la EV1, donde ya habíamos limpiado y transformado los datos. En esta etapa, el foco está en entrenar modelos de aprendizaje supervisado y no supervisado, optimizar hiperparámetros y analizar los resultados, todo orquestado con **Kedro**.

Los datos vienen de una empresa ficticia y están divididos en 4 archivos:

- `empleados.csv` — Información base de cada trabajador
- `evaluaciones.csv` — Evaluaciones de desempeño por periodo
- `capacitaciones.csv` — Cursos y horas de formación
- `ausencias.csv` — Registro de ausencias laborales

---

## Qué busca aprender el modelo

La idea principal del proyecto es ver si es posible predecir el desempeño de un empleado a partir de sus características. Para eso planteamos dos problemas distintos:

### Clasificación — ¿Este empleado tendrá alto desempeño?

Acá el modelo intenta aprender a distinguir entre empleados de **alto desempeño** (`alto_desempeno = 1`) y los que no (`= 0`), mirando variables como cuántas horas de capacitación tiene, cuántos días faltó, su antigüedad, el departamento donde trabaja y cómo le fue en evaluaciones anteriores.

La idea es que si el modelo aprende bien ese patrón, en la práctica podría ayudar a tomar decisiones sobre a quién retener, promover o apoyar con más formación.

> El target lo construimos comparando el `score_global` de cada empleado contra la mediana del grupo. Si está por encima, es alto desempeño.

### Regresión — ¿Cuánto va a rendir este empleado?

En vez de solo decir "alto o bajo", acá intentamos predecir directamente el número: el `avg_desempeno` (el promedio de sus evaluaciones de desempeño).

Es un problema más difícil porque tiene que estimar un valor continuo, pero también es más rico en información porque no reduce todo a una sola categoría.

---

## Contexto y limitaciones del dataset

Uno de los desafíos más grandes de este proyecto fue que el dataset es bastante pequeño. Con pocos registros, los modelos tienen menos ejemplos para aprender y es más fácil que se sobreajusten o que simplemente no logren capturar los patrones. Eso explica por qué el R² de regresión quedó en 0.2795, que no es el mejor resultado, pero es lo que el dataset nos permite con las herramientas que aplicamos.

### Cómo cuidamos los datos para no perder registros

Como teníamos pocos datos, no podíamos darnos el lujo de eliminar filas con valores faltantes. En cambio, usamos estas estrategias:

- **KNNImputer (k=5)**: para rellenar los valores faltantes en métricas de desempeño (`avg_desempeno`, `avg_tecnicas`, `avg_blandas`, `score_global`). Lo que hace es buscar los 5 empleados más parecidos y usar sus valores para estimar el que falta. Así conservamos cada registro en lugar de tirarlo a la basura.
- **Imputación con cero por lógica de negocio**: para ausencias y horas de capacitación, si no había registro simplemente pusimos 0. Tiene sentido porque si no hay registro, probablemente es que el empleado no faltó ni tomó cursos.

### Qué hicimos para que los modelos aprendieran mejor

- **`class_weight='balanced'`**: cuando las clases están desbalanceadas (hay muchos más empleados de un tipo que del otro), el modelo tiende a ignorar la clase minoritaria. Con esto lo forzamos a prestarle atención a las dos.
- **Modelos de ensemble** (RandomForest, GradientBoosting, XGBoost, StackingEnsemble): en vez de apostar todo a un solo modelo, estos combinan muchos modelos más simples. Con pocos datos funcionan mucho mejor porque reducen el sobreajuste.
- **Validación cruzada 5-fold**: en lugar de hacer una sola división train/test, dividimos el dataset en 5 partes y evaluamos el modelo 5 veces distintas. Eso da métricas mucho más confiables cuando los datos son escasos.
- **Split estratificado** (`stratify=y`): nos aseguramos de que tanto el train como el test tengan la misma proporción de clases. Con pocos datos, sin esto fácilmente el test puede quedar sin representantes de alguna clase.
- **Regularización** (Ridge, Lasso): básicamente le ponemos un límite a qué tan complejo puede ser el modelo, para que no memorice el dataset de entrenamiento.
- **Feature engineering**: creamos las variables `ratio_cap_antiguedad` y `tasa_ausencia`, que relacionan la capacitación y las ausencias con la antigüedad del empleado. Son variables nuevas que le dan más información al modelo sin necesitar más datos.

---

## Resultados principales

| Tarea | Mejor modelo | Métrica |
|---|---|---|
| Clasificación (alto desempeño) | XGBoost | F1 = 0.8148 |
| Regresión (puntaje de desempeño) | XGBoostRegressor | R² = 0.2795 |
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
│   ├── 01_exploratory_analysis.ipynb        → EDA inicial
│   ├── 02_supervised_modeling.ipynb         → Entrenamiento de modelos
│   ├── 03_model_evaluation.ipynb            → Evaluación y comparación
│   ├── 04_hyperparameter_optimization.ipynb → GridSearchCV / RandomizedSearchCV
│   └── 05_final_analysis.ipynb              → Análisis final integrado
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

O si querés también las herramientas de desarrollo:

```bash
pip install -e ".[dev]"
```

### 4. Agregar los CSV originales

Poner los 4 archivos en `data/01_raw/`:

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

Carga los 4 CSVs y genera un reporte de diagnóstico inicial para ver el estado de los datos antes de cualquier transformación.

**Output:** `data/08_reporting/reporte_diagnostico.csv`

### Pipeline 2 — Data Cleaning

Limpieza de los 4 datasets por separado. Se eliminan duplicados, se tratan los nulos, se corrigen tipos de datos y fechas, y se manejan los outliers con el método IQR.

**Output:** `data/02_intermediate/*_clean.csv`

### Pipeline 3 — Data Transformation

Se integran los 4 datasets en uno solo haciendo join por `id_empleado`. También se crean variables nuevas como `antiguedad_anos` y `score_global`, y se aplica normalización y codificación de variables categóricas.

**Output:** `data/03_primary/dataset_final_integrado.csv`

### Pipeline 4 — Data Validation

Verifica que todo el proceso de limpieza y transformación haya salido bien, comparando el antes y después.

**Output:** `data/08_reporting/reporte_comparacion.csv`, `reporte_validacion_final.csv`

### Pipeline 5 — Supervised Modeling

Entrena 13 modelos en total (7 de clasificación + 6 de regresión). Todos usan un Pipeline de Scikit-learn con StandardScaler incluido, y `random_state=42` para reproducibilidad.

**Modelos de clasificación** (`target: alto_desempeno`):

| Modelo | Configuración |
|---|---|
| LogisticRegression | `class_weight='balanced'` |
| DecisionTree | `class_weight='balanced'` |
| RandomForest | `n_estimators=100`, `class_weight='balanced'` |
| GradientBoosting | `n_estimators=100`, `learning_rate=0.1` |
| SVM | `probability=True`, `class_weight='balanced'` |
| XGBoost | `n_estimators=100` |
| StackingEnsemble | XGB + GB + SVM → LogisticRegression |

**Modelos de regresión** (`target: avg_desempeno`):

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

Acá aplicamos aprendizaje no supervisado para ver si hay grupos naturales entre los empleados sin usar etiquetas:

- **PCA**: reducimos las dimensiones a 2 componentes principales para poder visualizar los datos.
- **K-Means**: agrupamos los empleados en 3 clusters con `random_state=42`.

**Output:** `data/07_model_output/pca_resultado.csv`, `kmeans_resultado.csv`

### Pipeline 7 — Model Evaluation

Evaluamos todos los modelos con validación cruzada de 5 folds para tener métricas más robustas:

- Clasificación: F1, Accuracy, ROC-AUC
- Regresión: R², MAE, MSE, RMSE

**Output:** `data/07_model_output/cv_clasificacion.csv`, `cv_regresion.csv`

### Pipeline 8 — Hyperparameter Optimization

Buscamos los mejores hiperparámetros para los modelos más prometedores usando dos estrategias:

- **GridSearchCV**: prueba todas las combinaciones posibles de parámetros.
- **RandomizedSearchCV**: prueba combinaciones al azar, más rápido cuando el espacio es grande.

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
