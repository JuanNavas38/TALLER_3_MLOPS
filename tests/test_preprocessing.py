import pandas as pd
import pytest

from src.preprocessing import fit_encoders, encode_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo"],
        "island": ["Torgersen", "Dream", "Biscoe"],
        "bill_length_mm": [39.1, 46.5, 47.6],
        "bill_depth_mm": [18.7, 17.9, 14.6],
        "flipper_length_mm": [181.0, 192.0, 211.0],
        "body_mass_g": [3750.0, 3500.0, 4600.0],
        "sex": ["MALE", "FEMALE", "MALE"],
    })


def test_fit_encoders_contains_all_keys(sample_df):
    encoders = fit_encoders(sample_df)
    assert "island" in encoders
    assert "sex" in encoders
    assert "species" in encoders


def test_fit_encoders_classes(sample_df):
    encoders = fit_encoders(sample_df)
    assert set(encoders["island"].classes_) == {"Biscoe", "Dream", "Torgersen"}
    assert set(encoders["sex"].classes_) == {"FEMALE", "MALE"}
    assert set(encoders["species"].classes_) == {"Adelie", "Chinstrap", "Gentoo"}


def test_encode_features_converts_categoricals(sample_df):
    encoders = fit_encoders(sample_df)
    encoded = encode_features(sample_df, encoders)
    assert encoded["island"].dtype != object
    assert encoded["sex"].dtype != object
    assert encoded["species"].dtype != object


def test_encode_features_preserves_numerics(sample_df):
    encoders = fit_encoders(sample_df)
    encoded = encode_features(sample_df, encoders)
    assert list(encoded["bill_length_mm"]) == [39.1, 46.5, 47.6]
    assert list(encoded["body_mass_g"]) == [3750.0, 3500.0, 4600.0]


def test_encode_features_does_not_mutate_original(sample_df):
    encoders = fit_encoders(sample_df)
    original_island = list(sample_df["island"])
    encode_features(sample_df, encoders)
    assert list(sample_df["island"]) == original_island


def test_dropna_removes_incomplete_rows():
    df = pd.DataFrame({
        "species": ["Adelie", None, "Gentoo"],
        "island": ["Torgersen", "Dream", "Biscoe"],
        "bill_length_mm": [39.1, None, 47.6],
        "bill_depth_mm": [18.7, 17.9, 14.6],
        "flipper_length_mm": [181.0, 192.0, 211.0],
        "body_mass_g": [3750.0, 3500.0, 4600.0],
        "sex": ["MALE", "FEMALE", "MALE"],
    })
    cleaned = df.dropna()
    assert len(cleaned) == 2
    assert cleaned["species"].isna().sum() == 0


def test_encoders_fit_only_on_train_avoids_leakage():
    """Encoders trained only on train set must handle unseen values with an error."""
    train_df = pd.DataFrame({
        "species": ["Adelie", "Chinstrap"],
        "island": ["Torgersen", "Dream"],
        "bill_length_mm": [39.1, 46.5],
        "bill_depth_mm": [18.7, 17.9],
        "flipper_length_mm": [181.0, 192.0],
        "body_mass_g": [3750.0, 3500.0],
        "sex": ["MALE", "FEMALE"],
    })
    test_df_unseen = pd.DataFrame({
        "species": ["Gentoo"],  # Not seen during training
        "island": ["Biscoe"],
        "bill_length_mm": [47.6],
        "bill_depth_mm": [14.6],
        "flipper_length_mm": [211.0],
        "body_mass_g": [4600.0],
        "sex": ["MALE"],
    })

    encoders = fit_encoders(train_df)
    with pytest.raises(ValueError):
        encode_features(test_df_unseen, encoders)
