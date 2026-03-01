import json
import logging

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DROP_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    SVM_PARAMS,
    TEST_SIZE,
)
from src.preprocessing import (
    encode_features,
    fit_encoders,
    load_and_clean_data,
    save_encoders,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = load_and_clean_data()

    encoders = fit_encoders(df)
    df = encode_features(df, encoders)

    X = df.drop(columns=DROP_COLUMNS)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Random Forest
    logger.info("Entrenando Random Forest...")
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    logger.info("Accuracy Random Forest: %.4f", rf_accuracy)

    # SVM con Pipeline (incluye escalado)
    logger.info("Entrenando SVM con StandardScaler...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**SVM_PARAMS)),
    ])
    svm_pipeline.fit(X_train, y_train)
    y_pred_svm = svm_pipeline.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    logger.info("Accuracy SVM: %.4f", svm_accuracy)

    # Guardar artefactos
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(svm_pipeline, MODELS_DIR / "svm_model.pkl")
    save_encoders(encoders)

    # Guardar métricas
    metrics = {
        "rf": {
            "accuracy": rf_accuracy,
            "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
        },
        "svm": {
            "accuracy": svm_accuracy,
            "classification_report": classification_report(y_test, y_pred_svm, output_dict=True),
        },
    }
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas guardadas en %s", metrics_path)

    logger.info("Entrenamiento finalizado y artefactos guardados.")


if __name__ == "__main__":
    main()
