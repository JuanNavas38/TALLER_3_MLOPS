import logging
import sys

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    API_HOST,
    API_PORT,
    FEATURE_COLUMNS,
    FEATURE_RANGES,
    MODELS_DIR,
    ModelType,
)
from src.preprocessing import load_encoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Penguin Species Prediction API",
    description="API for predicting penguin species using MLOps pipeline",
)

# Cargar modelos y encoders al iniciar
try:
    rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
    encoders = load_encoders()
    logger.info("Modelos y encoders cargados correctamente.")
except FileNotFoundError as e:
    logger.error("Modelos no encontrados. Ejecuta primero: python -m src.train")
    logger.error("Detalle: %s", e)
    sys.exit(1)

MODELS = {
    ModelType.RF: rf_model,
    ModelType.SVM: svm_model,
}


class PenguinInput(BaseModel):
    island: str
    bill_length_mm: float = Field(ge=FEATURE_RANGES["bill_length_mm"][0], le=FEATURE_RANGES["bill_length_mm"][1])
    bill_depth_mm: float = Field(ge=FEATURE_RANGES["bill_depth_mm"][0], le=FEATURE_RANGES["bill_depth_mm"][1])
    flipper_length_mm: float = Field(ge=FEATURE_RANGES["flipper_length_mm"][0], le=FEATURE_RANGES["flipper_length_mm"][1])
    body_mass_g: float = Field(ge=FEATURE_RANGES["body_mass_g"][0], le=FEATURE_RANGES["body_mass_g"][1])
    sex: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Penguin Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": list(MODELS.keys())}


@app.post("/predict/{model_type}")
def predict_species(model_type: ModelType, penguin: PenguinInput):
    le_island = encoders["island"]
    le_sex = encoders["sex"]
    le_species = encoders["species"]

    if penguin.island not in le_island.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Island '{penguin.island}' desconocida. Opciones: {list(le_island.classes_)}",
        )
    if penguin.sex not in le_sex.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Sex '{penguin.sex}' desconocido. Opciones: {list(le_sex.classes_)}",
        )

    island_encoded = le_island.transform([penguin.island])[0]
    sex_encoded = le_sex.transform([penguin.sex])[0]

    data = pd.DataFrame(
        [[island_encoded, penguin.bill_length_mm, penguin.bill_depth_mm,
          penguin.flipper_length_mm, penguin.body_mass_g, sex_encoded]],
        columns=FEATURE_COLUMNS,
    )

    model = MODELS[model_type]
    try:
        prediction_idx = model.predict(data)[0]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Error en predicción: {e}")

    prediction_name = le_species.inverse_transform([prediction_idx])[0]

    logger.info("Predicción con %s: %s", model_type.value, prediction_name)
    return {"model_used": model_type.value, "prediction": prediction_name}


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
