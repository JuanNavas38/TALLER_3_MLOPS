import json
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure /app is in sys.path so we can import src modules
if '/app' not in sys.path:
    sys.path.append('/app')

MYSQL_USER = os.environ.get("MYSQL_USER", "myuser")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "mypassword")
MYSQL_CONN = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@mysql_db:3306/penguins_db"


def clear_db():
    engine = create_engine(MYSQL_CONN)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS raw_penguins"))
        conn.execute(text("DROP TABLE IF EXISTS preprocessed_penguins"))


def load_data():
    df = pd.read_csv('/app/data/penguins.csv')
    engine = create_engine(MYSQL_CONN)
    df.to_sql('raw_penguins', con=engine, if_exists='replace', index=False)


def preprocess_data():
    from src.config import MIN_SAMPLES_AFTER_CLEANING

    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql_table('raw_penguins', con=engine)

    df = df.dropna()
    if len(df) < MIN_SAMPLES_AFTER_CLEANING:
        raise ValueError(f"Too few samples after cleaning: {len(df)}")

    # Save clean (NOT encoded) data — encoders will be fit on train split only
    df.to_sql('preprocessed_penguins', con=engine, if_exists='replace', index=False)


def train_model():
    import mlflow
    import mlflow.sklearn
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from src.config import DROP_COLUMNS, TEST_SIZE, RANDOM_STATE, RF_PARAMS, SVM_PARAMS, MODELS_DIR
    from src.preprocessing import fit_encoders, encode_features, save_encoders

    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql_table('preprocessed_penguins', con=engine)

    # Split BEFORE encoding to prevent data leakage into encoders
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["species"]
    )

    # Fit encoders on training data ONLY
    encoders = fit_encoders(train_df)
    save_encoders(encoders)

    train_enc = encode_features(train_df, encoders)
    test_enc = encode_features(test_df, encoders)

    X_train = train_enc.drop(columns=DROP_COLUMNS)
    y_train = train_enc["species"]
    X_test = test_enc.drop(columns=DROP_COLUMNS)
    y_test = test_enc["species"]

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("penguins_classification")

    run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # --- Random Forest ---
        rf_model = RandomForestClassifier(**RF_PARAMS)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        rf_report = classification_report(y_test, rf_preds, output_dict=True)
        print(f"Random Forest Accuracy: {rf_acc}")

        mlflow.log_params({f"rf_{k}": v for k, v in RF_PARAMS.items()})
        mlflow.log_metric("rf_accuracy", rf_acc)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")

        # --- SVM ---
        svm_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(**SVM_PARAMS)),
        ])
        svm_pipeline.fit(X_train, y_train)
        svm_preds = svm_pipeline.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_preds)
        svm_report = classification_report(y_test, svm_preds, output_dict=True)
        print(f"SVM Accuracy: {svm_acc}")

        mlflow.log_params({f"svm_{k}": v for k, v in SVM_PARAMS.items()})
        mlflow.log_metric("svm_accuracy", svm_acc)
        mlflow.sklearn.log_model(svm_pipeline, "svm_model")

    # Save model artifacts locally for the API to consume
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(svm_pipeline, MODELS_DIR / "svm_model.pkl")

    # Save metrics summary to disk
    metrics = {
        "trained_at": datetime.now().isoformat(),
        "rf": {"accuracy": rf_acc, "classification_report": rf_report},
        "svm": {"accuracy": svm_acc, "classification_report": svm_report},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


default_args = {
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False
}

with DAG(
    dag_id="penguins_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1
) as dag:

    t1 = PythonOperator(
        task_id="clear_db",
        python_callable=clear_db
    )

    t2 = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    t3 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    t4 = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    t1 >> t2 >> t3 >> t4
