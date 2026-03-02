# Taller 3: Palmer Penguins - MLOps con Airflow

**MLOps - Pontificia Universidad Javeriana**

**Pipeline automatizado de entrenamiento usando Apache Airflow, MySQL, PostgreSQL y Docker Compose.**

## Autores
*   **Juan Navas**
*   **Camila Cuellar**
*   **Jhonathan Murcia**

---

## Descripción General
Este proyecto evoluciona el Taller 2 agregando **Apache Airflow** como orquestador del pipeline de ML. 
El sistema usa dos bases de datos separadas:
- **MySQL**: exclusiva para datos de pingüinos (raw y procesados)
- **PostgreSQL**: exclusiva para metadatos de Airflow

El DAG automatiza todo el flujo desde la carga de datos hasta el entrenamiento del modelo.

---

## Arquitectura

```
MySQL (penguins_db)  →  Datos raw y procesados de pingüinos
PostgreSQL           →  Metadatos internos de Airflow
Airflow Scheduler    →  Ejecuta el DAG automáticamente
Airflow Webserver    →  UI visual en puerto 8080
API FastAPI          →  Inferencia del modelo entrenado
```

---

## Estructura del Proyecto

```text
TALLER_3_MLOPS/
├── src/                                # Código fuente
│   ├── config.py                       # Configuración (paths, DB, modelos)
│   ├── database.py                     # Conexión y operaciones MySQL
│   ├── preprocessing.py                # Limpieza y encoding
│   ├── train.py                        # Entrenamiento
│   └── api.py                          # FastAPI inferencia
├── dags/                               # DAGs de Airflow
│   └── penguin_training_dag.py         # Pipeline completo
├── data/                               # Dataset local (penguins.csv)
├── models/                             # Artefactos (PKL, metrics.json)
├── docker-compose.yml                  # Orquestación de todos los servicios
├── Dockerfile                          # Imagen de la API
├── pyproject.toml                      # Gestión de dependencias con uv
├── .env                                # Variables de entorno
└── README.md
```

---

## Pipeline del DAG

El DAG `penguin_training_pipeline` ejecuta 5 tareas en orden:

```
[1. clear_database] → [2. load_raw_data] → [3. preprocess_data] → [4. train_model] → [5. verify_artifacts]
```

| Tarea | Descripción |
|-------|-------------|
| `clear_database` | Borra el contenido de las tablas en MySQL |
| `load_raw_data` | Carga el CSV a MySQL **sin preprocesamiento** |
| `preprocess_data` | Limpia, codifica y guarda datos procesados en MySQL |
| `train_model` | Lee datos procesados de MySQL y entrena RF + SVM |
| `verify_artifacts` | Verifica que los modelos y métricas fueron guardados |

---

## Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Airflow
AIRFLOW_UID=50000

# MySQL - Base de datos de datos
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=penguin_user
MYSQL_PASSWORD=penguin_pass
MYSQL_DATABASE=penguins_db
MYSQL_ROOT_PASSWORD=root_pass

# PostgreSQL - Metadatos de Airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
```

---

## Ejecución con Docker Compose

### 1. Levantar todos los servicios
```powershell
docker-compose up --build -d
```

Esto iniciará en orden:
1. **MySQL** y **PostgreSQL** (bases de datos)
2. **Airflow Init** (inicializa DB y crea usuario admin)
3. **Airflow Scheduler** y **Airflow Webserver**
4. **API FastAPI**

### 2. Verificar que todo está corriendo
```powershell
docker-compose ps
```

### 3. Acceder a la UI de Airflow
- URL: [http://localhost:8080](http://localhost:8080)
- Usuario: `admin`
- Contraseña: `admin`

### 4. Ejecutar el DAG
1. Abre [http://localhost:8080](http://localhost:8080)
2. Busca el DAG `penguin_training_pipeline`
3. Actívalo con el toggle ▶️
4. Haz clic en **Trigger DAG** para ejecutarlo manualmente

### 5. Predicciones vía API
Una vez entrenado el modelo, puedes hacer predicciones:

```powershell
curl -X POST "http://localhost:8989/predict/rf" `
     -H "Content-Type: application/json" `
     -d '{
           "island": "Biscoe",
           "bill_length_mm": 45.0,
           "bill_depth_mm": 15.0,
           "flipper_length_mm": 210.0,
           "body_mass_g": 4000.0,
           "sex": "male"
         }'
```

- **`/predict/rf`** → Random Forest
- **`/predict/svm`** → Support Vector Machine
- **Docs**: [http://localhost:8989/docs](http://localhost:8989/docs)

### 6. Detener los servicios
```powershell
docker-compose down
```

Para eliminar también los volúmenes (borra datos de BD):
```powershell
docker-compose down -v
```

---

## Servicios y Puertos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| Airflow UI | 8080 | Panel de control del pipeline |
| API FastAPI | 8989 | Inferencia del modelo |
| MySQL | 3306 | Base de datos de pingüinos |
| PostgreSQL | 5432 | Metadatos de Airflow |

---

## Desarrollo Local (Opcional)
Si tienes `uv` instalado localmente:
```bash
uv sync
uv run python -m src.train
uv run python -m src.api
```