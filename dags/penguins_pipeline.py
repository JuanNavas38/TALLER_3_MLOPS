import sys
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure /app is in sys.path so we can import src modules
if '/app' not in sys.path:
    sys.path.append('/app')

MYSQL_CONN = "mysql+pymysql://myuser:mypassword@mysql_db:3306/penguins_db"

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
    from src.preprocessing import fit_encoders, encode_features, save_encoders
    from src.config import MIN_SAMPLES_AFTER_CLEANING
    
    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql_table('raw_penguins', con=engine)
    
    df = df.dropna()
    if len(df) < MIN_SAMPLES_AFTER_CLEANING:
        raise ValueError(f"Too few samples: {len(df)}")
        
    encoders = fit_encoders(df)
    save_encoders(encoders)
    
    df_encoded = encode_features(df, encoders)
    df_encoded.to_sql('preprocessed_penguins', con=engine, if_exists='replace', index=False)

def train_model():
    from src.config import DROP_COLUMNS, TEST_SIZE, RANDOM_STATE, RF_PARAMS, SVM_PARAMS, MODELS_DIR
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import joblib
    
    engine = create_engine(MYSQL_CONN)
    df = pd.read_sql_table('preprocessed_penguins', con=engine)
    
    X = df.drop(columns=DROP_COLUMNS)
    y = df["species"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy: {rf_acc}")
    
    # Train SVM
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**SVM_PARAMS)),
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_pipeline.predict(X_test))
    print(f"SVM Accuracy: {svm_acc}")
    
    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(svm_pipeline, MODELS_DIR / "svm_model.pkl")

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
