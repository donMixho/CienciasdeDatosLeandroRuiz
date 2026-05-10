# 🧠 LeandroCienciasDeDatos — EV1

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)

## 📋 Descripción

Proyecto de transformación de datos del caso **Recursos Humanos**, desarrollado como parte de la asignatura **SCY1101 - Programación para la Ciencia de Datos**. Simula un entorno profesional de ciencia de datos utilizando el framework **Kedro** como orquestador del flujo de datos.

Los datos corresponden a una empresa con 4 datasets relacionados:

- `empleados.csv` — Información base de cada trabajador
- `evaluaciones.csv` — Evaluaciones de desempeño por periodo
- `capacitaciones.csv` — Cursos y horas de formación
- `ausencias.csv` — Registro de ausencias laborales

---

---

## 🏗️ Estructura del Proyecto

CienciasdeDatosLeandroRuiz/
├── conf/
│ └── base/
│ ├── catalog.yml → Definición de datasets
│ └── parameters.yml → Parámetros configurables
├── data/
│ ├── 01_raw/ → CSVs originales (no modificar)
│ ├── 02_intermediate/ → Datos limpios
│ ├── 03_primary/ → Dataset final integrado
│ └── 08_reporting/ → Reportes de diagnóstico y validación
├── notebooks/
│ └── Ciencias_De_Datos.ipynb → EDA exploratorio
├── src/leandrocienciasdedatos/
│ └── pipelines/
│ ├── data_ingestion/ → Pipeline 1: Carga y diagnóstico
│ ├── data_cleaning/ → Pipeline 2: Limpieza
│ ├── data_transformation/ → Pipeline 3: Transformación
│ └── data_validation/ → Pipeline 4: Validación
├── requirements.txt
└── README.md

---

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/donMixho/CienciasdeDatosLeandroRuiz.git
cd CienciasdeDatosLeandroRuiz
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

### 3. Activar entorno virtual

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Agregar los CSV originales

Coloca los 4 archivos en `data/01_raw/`:

- `empleados.csv`
- `evaluaciones.csv`
- `capacitaciones.csv`
- `ausencias.csv`

---

## 🚀 Ejecución

### Correr todos los pipelines

```bash
kedro run
```

### Correr un pipeline específico

```bash
kedro run --pipeline data_ingestion
kedro run --pipeline data_cleaning
kedro run --pipeline data_transformation
kedro run --pipeline data_validation
```

---

## 🔄 Pipelines

### Pipeline 1 — Data Ingestion

Carga los 4 CSVs y genera un reporte de diagnóstico inicial con métricas de calidad de datos.

**Output:** `data/08_reporting/reporte_diagnostico.csv`

### Pipeline 2 — Data Cleaning

Limpia los 4 datasets aplicando:

- Eliminación de duplicados
- Tratamiento de valores nulos
- Corrección de tipos de datos
- Estandarización de fechas con formato mixto
- Normalización de strings
- Tratamiento de outliers con método IQR

**Output:** `data/02_intermediate/*_clean.csv`

### Pipeline 3 — Data Transformation

Integra los 4 datasets limpios en uno consolidado aplicando:

- Joins/merges por `id_empleado`
- Aggregations con `groupby`
- Features derivadas: `antiguedad_anos`, `score_global`
- Normalización con `MinMaxScaler`
- Codificación de variables categóricas con `LabelEncoder`

**Output:** `data/03_primary/dataset_final_integrado.csv`

### Pipeline 4 — Data Validation

Verifica la integridad del proceso completo:

- Comparación antes/después de limpieza
- Validación de esquema del dataset final
- Verificación de columnas clave sin nulos

**Output:** `data/08_reporting/reporte_comparacion.csv`, `data/08_reporting/reporte_validacion_final.csv`

---

## 📦 Dependencias principales

| Librería              | Uso                             |
| --------------------- | ------------------------------- |
| `kedro~=1.3.1`        | Orquestación del flujo de datos |
| `pandas>=2.3.3`       | Manipulación de datos           |
| `numpy>=1.26.0`       | Operaciones numéricas           |
| `scikit-learn>=1.3.0` | Normalización y encoding        |

---

## 👤 Autor

**Leandro Ruiz**  
Ingeniería Informática — Ciencia de Datos  
SCY1101 — 2025
