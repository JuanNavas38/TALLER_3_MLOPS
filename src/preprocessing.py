import logging
from typing import Dict

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import CATEGORICAL_COLUMNS, DATA_PATH, MIN_SAMPLES_AFTER_CLEANING, MODELS_DIR

logger = logging.getLogger(__name__)


def load_and_clean_data() -> pd.DataFrame:
    logger.info("Cargando datos desde %s...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    original_shape = df.shape
    logger.info("Dimensiones originales: %s", original_shape)

    df = df.dropna()
    logger.info("Dimensiones tras eliminar nulos: %s", df.shape)

    if len(df) < MIN_SAMPLES_AFTER_CLEANING:
        raise ValueError(
            f"Solo quedan {len(df)} muestras tras limpieza "
            f"(mínimo requerido: {MIN_SAMPLES_AFTER_CLEANING})"
        )

    return df


def fit_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
    # Encoder para el target
    le_species = LabelEncoder()
    le_species.fit(df["species"])
    encoders["species"] = le_species
    return encoders


def encode_features(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = encoders[col].transform(df[col])
    if "species" in df.columns:
        df["species"] = encoders["species"].transform(df["species"])
    return df


def save_encoders(encoders: Dict[str, LabelEncoder]) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, encoder in encoders.items():
        path = MODELS_DIR / f"label_encoder_{name}.pkl"
        joblib.dump(encoder, path)
        logger.info("Encoder guardado: %s", path)


def load_encoders() -> Dict[str, LabelEncoder]:
    encoders = {}
    for name in [*CATEGORICAL_COLUMNS, "species"]:
        path = MODELS_DIR / f"label_encoder_{name}.pkl"
        encoders[name] = joblib.load(path)
    return encoders
