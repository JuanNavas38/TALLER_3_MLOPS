# Taller 3: Palmer Penguins - Airflow & MLOps

**MLOps - Pontificia Universidad Javeriana**

**Orquestación de pipeline de Machine Learning usando Apache Airflow, Docker Compose y `uv`.**

## Autores
*   **Juan Navas**
*   **Camila Cuellar**
*   **Jhonathan Murcia**

---

## Descripción General
Este proyecto evoluciona la arquitectura anterior integrando **Apache Airflow** para orquestar de manera automática el pipeline completo de machine learning. Todos los recursos están contenedorizados y manejados mediante un único archivo `docker-compose.yml`.

Se han eliminado componentes interactivos (Jupyter Notebooks) y scripts standalone de entrenamiento para darle paso a un DAG centralizado que carga, preprocesa y entrena los modelos automáticamente sobre una base de datos **MySQL**, dejando los artefactos listos para el consumo mediante una **API REST (FastAPI)**.

## Arquitectura y Flujo de Trabajo

1. **MySQL**: Base de datos dedicada exclusivamente para almacenar los datos (crudos y preprocesados) de los pingüinos.
2. **Airflow**: 
   - Consume los datos base de `data/penguins.csv`.
   - Limpia y carga la información cruda a MySQL (`raw_penguins`).
   - Preprocesa la información y la guarda de nuevo en MySQL (`preprocessed_penguins`).
   - Entrena los modelos (Random Forest y SVM) usando la base de datos MySQL, y guarda los artefactos en el volumen compartido `models/`.
3. **API (FastAPI)**: Cuando el pipeline de Airflow finaliza, la API lee los artefactos desde `models/` y queda disponible para realizar la inferencia.

---

## Estructura del Proyecto

```text
TALLER_3_MLOPS/
├── dags/                               # DAGs de Airflow
│   └── penguins_pipeline.py            # DAG principal del pipeline
├── src/                                # Código fuente secundario
│   ├── config.py                       # Parámetros compartidos
│   ├── preprocessing.py                # Lógica de codificación usada por el DAG
│   └── api.py                          # FastAPI para inferencias
├── data/                               # Dataset base local (penguins.csv)
├── models/                             # Artefactos generados compartidos (PKL)
├── docker-compose.yml                  # Orquestación de absolutamente todos los servicios
├── Dockerfile                          # Imagen base para la API con uv
├── pyproject.toml                      # Gestión de dependencias de python (uv)
├── .env                                # Configuración segura (e.g., AIRFLOW_UID)
└── README.md
```

*(Nota: Archivos en desuso como el antiguo entorno Jupyter, el Dockerfile.jupyter, y el standalone `train.py` han sido removidos deliberadamente para centralizar el flujo en Airflow).*

---

## Instrucciones de Uso

### 1. Iniciar los Servicios
Para desplegar todos los servicios conectados (Airflow Webserver, Scheduler, Celery Workers, Redis, Postgres para la metadata de Airflow, MySQL para datos de negocio y la API de predicciones):

```powershell
docker compose up -d
```

### 2. Verificar Airflow y Ejecutar el DAG
1. Dirígete a tu navegador web e ingresa al Airflow Webserver: **[http://localhost:8080](http://localhost:8080)**.
2. Ingresa con las credenciales por defecto:
   - **Usuario**: `airflow`
   - **Contraseña**: `airflow`
3. En la lista de DAGs, busca el DAG llamado **`penguins_pipeline`**.
4. Actívalo temporalmente (Trigger) usando el botón circular de "Play", en la sección Actions.
5. Puedes hacer clic en el DAG y revisar cómo los "Tasks" cambian a verde claro (corriendo) y luego a verde oscuro (completado exitosamente). Las tareas son: `clear_db` -> `load_data` -> `preprocess_data` -> `train_model`.

### 3. Verificar y Consumir la API
Una vez que el DAG aparezca como `success` en todos sus pasos, significará que los modelos han sido entrenados satisfactoriamente y guardados en el contenedor.

Puedes revisar y probar el estado de la inferencia consultando la API generada.

*   **Documentación Interactiva (Swagger)**: [http://localhost:8989/docs](http://localhost:8989/docs)
*   **Health Check**: Verifica si los modelos subieron correctamente: [http://localhost:8989/health](http://localhost:8989/health)

**Ejemplo de Petición cURL:**
```powershell
curl -X POST "http://localhost:8989/predict/rf" `
     -H "Content-Type: application/json" `
     -d '{
           "island": "Biscoe",
           "bill_length_mm": 45.0,
           "bill_depth_mm": 15.0,
           "flipper_length_mm": 210.0,
           "body_mass_g": 4000.0,
           "sex": "MALE"
         }'
```
*(Cambia el endpoint `/predict/rf` por `/predict/svm` para evaluar el clasificador alternativo).*
