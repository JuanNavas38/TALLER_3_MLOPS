import json

import pytest

from src.config import MODELS_DIR
from src.train import main


@pytest.fixture(scope="module")
def trained_artifacts():
    """Run training once for all tests in this module."""
    main()
    return MODELS_DIR


def test_rf_model_saved(trained_artifacts):
    assert (trained_artifacts / "random_forest_model.pkl").exists()


def test_svm_model_saved(trained_artifacts):
    assert (trained_artifacts / "svm_model.pkl").exists()


def test_encoders_saved(trained_artifacts):
    assert (trained_artifacts / "label_encoder_island.pkl").exists()
    assert (trained_artifacts / "label_encoder_sex.pkl").exists()
    assert (trained_artifacts / "label_encoder_species.pkl").exists()


def test_metrics_saved(trained_artifacts):
    metrics_path = trained_artifacts / "metrics.json"
    assert metrics_path.exists()

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert "rf" in metrics
    assert "svm" in metrics
    assert metrics["rf"]["accuracy"] > 0.8
    assert metrics["svm"]["accuracy"] > 0.8
