from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "penguins.csv"

API_HOST = "0.0.0.0"
API_PORT = 8989

FEATURE_COLUMNS = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
CATEGORICAL_COLUMNS = ["island", "sex"]
TARGET_COLUMN = "species"
DROP_COLUMNS = ["species"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_PARAMS = {"n_estimators": 100, "random_state": RANDOM_STATE}
SVM_PARAMS = {"probability": True, "random_state": RANDOM_STATE}

MIN_SAMPLES_AFTER_CLEANING = 50

FEATURE_RANGES = {
    "bill_length_mm": (25.0, 65.0),
    "bill_depth_mm": (12.0, 22.0),
    "flipper_length_mm": (170.0, 240.0),
    "body_mass_g": (2500.0, 6500.0),
}


class ModelType(str, Enum):
    RF = "rf"
    SVM = "svm"
