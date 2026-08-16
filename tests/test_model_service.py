"""Tests für Modellvorhersage und Eingabevalidierung."""

import pytest

from src.model_service import predict_species


def test_predict_species_returns_valid_prediction():
    """Eine plausible Gentoo-Beobachtung soll korrekt verarbeitet werden."""

    observation = {
        "bill_length_mm": 47.5,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    result = predict_species(observation)

    assert result["predicted_species"] in {
        "Adelie",
        "Chinstrap",
        "Gentoo",
    }

    assert set(result["probabilities"].keys()) == {
        "Adelie",
        "Chinstrap",
        "Gentoo",
    }

    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9


def test_unknown_sex_is_allowed():
    """Unbekanntes Geschlecht soll als gültige Eingabe akzeptiert werden."""

    observation = {
        "bill_length_mm": 47.5,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    result = predict_species(observation)

    assert result["validated_observation"]["sex"] == "unknown"


def test_missing_numeric_value_raises_error():
    """Ein fehlender Körpermesswert soll einen verständlichen Fehler erzeugen."""

    observation = {
        "bill_length_mm": None,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    with pytest.raises(ValueError):
        predict_species(observation)


def test_out_of_training_range_creates_warning():
    """Ein extremer, aber technisch gültiger Wert soll nur gewarnt werden."""

    observation = {
        "bill_length_mm": 70.0,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    result = predict_species(observation)

    assert len(result["warnings"]) >= 1
    assert any(
        "außerhalb des Trainingsbereichs" in warning
        for warning in result["warnings"]
    )