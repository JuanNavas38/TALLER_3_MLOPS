import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Agregar /opt/airflow al path para importar src
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# -----------------------------------------------
# TAREA 1 - Borrar contenido de la base de datos
# -----------------------------------------------
def task_clear_database():
    from src.database import get_engine, create_tables, clear_tables

    logger.info("Iniciando limpieza de base de datos...")
    engine = get_engine()
    create_tables(engine)
    clear_tables(engine)
    logger.info("Base de datos limpia y lista.")


# -----------------------------------------------
# TAREA 2 - Cargar datos raw a MySQL sin preprocesamiento
# -----------------------------------------------
def task_load_raw_data():
    import pandas as pd
    from src.config import DATA_PATH
    from src.database import get_engine, load_raw_data

    logger.info("Cargando datos raw desde %s...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    original_shape = df.shape
    logger.info("Datos leídos: %s filas, %s columnas", *original_shape)

    engine = get_engine()
    load_raw_data(engine, df)
    logger.info("Datos raw cargados en MySQL: %d filas.", len(df))


# -----------------------------------------------
# TAREA 3 - Preprocesar datos y guardar en MySQL
# -----------------------------------------------
def task_preprocess_data():
    from src.database import get_engine, load_processed_data
    from src.preprocessing import (
        encode_features,
        fit_encoders,
        load_and_clean_data,
        save_encoders,
    )

    logger.info("Iniciando preprocesamiento...")

    # Cargar y limpiar datos desde el CSV original
    df = load_and_clean_data()
    logger.info("Datos limpios: %d filas.", len(df))

    # Ajustar encoders y transformar
    encoders = fit_encoders(df)
    save_encoders(encoders)
    df_encoded = encode_features(df, encoders)
    logger.info("Encoders guardados y features codificadas.")

    # Guardar datos procesados en MySQL
    engine = get_engine()
    load_processed_data(engine, df_encoded)
    logger.info("Datos procesados guardados en MySQL: %d filas.", len(df_encoded))


# -----------------------------------------------
# TAREA 4 - Entrenar modelos con datos de MySQL
# -----------------------------------------------
def task_train_model():
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    import joblib

    from src.config import (
        DROP_COLUMNS,
        MODELS_DIR,
        RANDOM_STATE,
        RF_PARAMS,
        SVM_PARAMS,
        TEST_SIZE,
    )
    from src.database import get_engine, read_processed_data

    logger.info("Leyendo datos procesados desde MySQL...")
    engine = get_engine()
    df = read_processed_data(engine)
    logger.info("Datos leídos para entrenamiento: %d filas.", len(df))

    # Separar features y target
    X = df.drop(columns=DROP_COLUMNS)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Random Forest ---
    logger.info("Entrenando Random Forest...")
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**RF_PARAMS)),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_pipeline.predict(X_test))
    logger.info("Random Forest accuracy: %.4f", rf_accuracy)

    # --- SVM ---
    logger.info("Entrenando SVM...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(**SVM_PARAMS)),
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_accuracy = accuracy_score(y_test, svm_pipeline.predict(X_test))
    logger.info("SVM accuracy: %.4f", svm_accuracy)

    # --- Guardar modelos ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(svm_pipeline, MODELS_DIR / "svm_model.pkl")
    logger.info("Modelos guardados en %s", MODELS_DIR)

    # --- Guardar métricas ---
    metrics = {
        "random_forest": {"accuracy": rf_accuracy},
        "svm": {"accuracy": svm_accuracy},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Métricas guardadas: %s", metrics)


# -----------------------------------------------
# TAREA 5 - Verificar que los artefactos existen
# -----------------------------------------------
def task_verify_artifacts():
    from src.config import MODELS_DIR

    artifacts = [
        "random_forest_model.pkl",
        "svm_model.pkl",
        "metrics.json",
    ]

    logger.info("Verificando artefactos en %s...", MODELS_DIR)
    for artifact in artifacts:
        path = MODELS_DIR / artifact
        if not path.exists():
            raise FileNotFoundError(f"Artefacto no encontrado: {path}")
        logger.info("✓ %s verificado.", artifact)

    logger.info("Todos los artefactos verificados correctamente.")


# -----------------------------------------------
# DEFINICIÓN DEL DAG
# -----------------------------------------------
with DAG(
    dag_id="penguin_training_pipeline",
    description="Pipeline completo de entrenamiento para clasificación de pingüinos",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["penguins", "mlops", "training"],
) as dag:

    clear_database = PythonOperator(
        task_id="clear_database",
        python_callable=task_clear_database,
    )

    load_raw_data = PythonOperator(
        task_id="load_raw_data",
        python_callable=task_load_raw_data,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    verify_artifacts = PythonOperator(
        task_id="verify_artifacts",
        python_callable=task_verify_artifacts,
    )

    # Orden de ejecución
    clear_database >> load_raw_data >> preprocess_data >> train_model >> verify_artifacts