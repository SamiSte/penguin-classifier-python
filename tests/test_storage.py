"""Tests für die Speicherung neuer Pinguinbeobachtungen."""

import pandas as pd

from src.storage import CSV_COLUMNS, save_observation


def test_save_observation_creates_csv(tmp_path):
    """Eine Beobachtung soll korrekt in einer neuen CSV gespeichert werden."""

    test_path = tmp_path / "new_observations.csv"

    observation = {
        "bill_length_mm": 47.5,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    probabilities = {
        "Adelie": 0.002,
        "Chinstrap": 0.000,
        "Gentoo": 0.998,
    }

    save_observation(
        observation=observation,
        island="Biscoe",
        predicted_species="Gentoo",
        probabilities=probabilities,
        output_path=test_path,
    )

    assert test_path.exists()

    saved_data = pd.read_csv(test_path)

    assert len(saved_data) == 1
    assert list(saved_data.columns) == CSV_COLUMNS
    assert saved_data.loc[0, "predicted_species"] == "Gentoo"
    assert saved_data.loc[0, "island"] == "Biscoe"
    assert saved_data.loc[0, "probability_gentoo"] == 0.998


def test_save_observation_appends_rows(tmp_path):
    """Mehrere Beobachtungen sollen angehängt und nicht überschrieben werden."""

    test_path = tmp_path / "new_observations.csv"

    observation = {
        "bill_length_mm": 47.5,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }

    probabilities = {
        "Adelie": 0.002,
        "Chinstrap": 0.000,
        "Gentoo": 0.998,
    }

    for _ in range(2):
        save_observation(
            observation=observation,
            island="Biscoe",
            predicted_species="Gentoo",
            probabilities=probabilities,
            output_path=test_path,
        )

    saved_data = pd.read_csv(test_path)

    assert len(saved_data) == 2