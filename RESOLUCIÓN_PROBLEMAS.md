# Resolución de Problemas - Pipeline de Transformación

## Problemas Encontrados y Soluciones Aplicadas

### 1. **Error: `CSVDataSet` no encontrado**

**Problema:** Kedro v1.3+ usa `CSVDataset` (minúscula 's'), no `CSVDataSet`

```
DatasetError: Dataset 'CSVDataSet' not found in 'pandas'
```

**Solución:**

- Actualizar `conf/base/catalog.yml`
- Cambiar `pandas.CSVDataSet` → `pandas.CSVDataset` en todos los datasets

---

### 2. **Error: Pipeline `data_transformation` no encontrado**

**Problema:** El pipeline existía pero no estaba registrado

**Solución:**

- En `src/leandrocienciasdedatos/pipeline_registry.py`:
  - Agregar import: `from leandrocienciasdedatos.pipelines.data_transformation.pipeline import create_pipeline as create_data_transformation_pipeline`
  - Registrar en el diccionario: `"data_transformation": create_data_transformation_pipeline()`

---

### 3. **Error: Problema de Codificación (UnicodeDecodeError)**

**Problema:** Los CSV están codificados en `latin-1`, no UTF-8

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfa in position 192
```

**Solución:**

- Actualizar `conf/base/catalog.yml` para especificar encoding en todos los datasets:

```yaml
dataset_name:
  type: pandas.CSVDataset
  filepath: data/.../archivo.csv
  load_args:
    encoding: latin-1
  save_args:
    encoding: latin-1
```

---

### 4. **Error: Parámetros mal configurados**

**Problema:** Intento de acceder a `parameters["transform"]["primary_key"]` cuando se pasaba `params:transform`

**Solución:**

- En `src/leandrocienciasdedatos/pipelines/data_transformation/nodes.py`:
  - Cambiar `parameters["transform"]["primary_key"]` → `parameters["primary_key"]`
  - Cambiar `parameters["transform"]["evaluation_metrics"]` → `parameters["evaluation_metrics"]`
- El pipeline pasa `params:transform` que contiene directamente las claves

---

## Pasos para Reproducir la Configuración Correcta

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
# O específicamente:
python -m pip install scikit-learn
```

### 2. Verificar Catálogo

- ✅ Todos los datasets usan `pandas.CSVDataset` (no CSVDataSet)
- ✅ Todos los datasets tienen `encoding: latin-1` en load_args/save_args

### 3. Verificar Pipeline Registry

- ✅ Todas las pipelines importadas están registradas
- ✅ Usa `create_pipeline` como función creadora

### 4. Verificar Parámetros

- ✅ `conf/base/parameters.yml` contiene las secciones `cleaning` y `transform`
- ✅ Los nodos acceden correctamente: `parameters["clave"]` (no anidado)

---

## Comandos para Ejecutar el Pipeline

```bash
# Pipeline de ingesta
kedro run --pipeline data_ingestion

# Pipeline de limpieza
kedro run --pipeline data_cleaning

# Pipeline de transformación
kedro run --pipeline data_transformation

# Todos los pipelines
kedro run
```

---

## Resumen de Cambios Realizados

| Archivo                                   | Cambio                                             |
| ----------------------------------------- | -------------------------------------------------- |
| `conf/base/catalog.yml`                   | Cambiar CSVDataSet → CSVDataset + agregar encoding |
| `src/.../pipeline_registry.py`            | Registrar pipeline data_transformation             |
| `src/.../data_transformation/nodes.py`    | Ajustar acceso a parámetros                        |
| `src/.../data_transformation/pipeline.py` | Usar `params:transform`                            |

---

## Estado Actual

✅ Todos los pipelines funcionan correctamente

- data_ingestion: ✅
- data_cleaning: ✅
- data_transformation: ✅
