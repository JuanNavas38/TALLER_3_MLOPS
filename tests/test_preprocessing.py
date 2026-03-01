import pandas as pd
import pytest

from src.preprocessing import encode_features, fit_encoders, load_and_clean_data


def test_load_and_clean_data_returns_dataframe():
    df = load_and_clean_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.isnull().sum().sum() == 0


def test_load_and_clean_data_has_expected_columns():
    df = load_and_clean_data()
    expected = {"species", "island", "bill_length_mm", "bill_depth_mm",
                "flipper_length_mm", "body_mass_g", "sex", "year"}
    assert expected == set(df.columns)


def test_fit_encoders_returns_all_encoders():
    df = load_and_clean_data()
    encoders = fit_encoders(df)
    assert "island" in encoders
    assert "sex" in encoders
    assert "species" in encoders


def test_fit_encoders_known_classes():
    df = load_and_clean_data()
    encoders = fit_encoders(df)
    assert "Biscoe" in encoders["island"].classes_
    assert "male" in encoders["sex"].classes_.tolist() or "Male" in encoders["sex"].classes_.tolist()
    assert len(encoders["species"].classes_) == 3


def test_encode_features_transforms_categoricals():
    df = load_and_clean_data()
    encoders = fit_encoders(df)
    encoded = encode_features(df, encoders)

    assert encoded["island"].dtype in ("int32", "int64")
    assert encoded["sex"].dtype in ("int32", "int64")
    assert encoded["species"].dtype in ("int32", "int64")
    # Numeric columns should remain unchanged
    assert encoded["bill_length_mm"].dtype == "float64"
