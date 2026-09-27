"""Tests für die Speicherung neuer Pinguinbeobachtungen."""

import pandas as pd
import pytest

from src.storage import ALLOWED_SPECIES, CSV_COLUMNS, save_observation


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
    assert pd.isna(saved_data.loc[0, "validated_species"])


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


@pytest.fixture
def observation():
    return {
        "bill_length_mm": 47.5,
        "bill_depth_mm": 14.5,
        "flipper_length_mm": 218.0,
        "body_mass_g": 5100.0,
        "sex": "unknown",
    }


@pytest.fixture
def probabilities():
    return {"Adelie": 0.002, "Chinstrap": 0.0, "Gentoo": 0.998}


@pytest.mark.parametrize("validated_species", [None, ""])
def test_unconfirmed_species_is_saved_as_empty_cell(
    tmp_path, observation, probabilities, validated_species
):
    """Eine Vorhersage allein liefert keine bestätigte Art."""
    test_path = tmp_path / "new_observations.csv"

    save_observation(
        observation,
        "Biscoe",
        "Gentoo",
        probabilities,
        test_path,
        validated_species=validated_species,
    )

    saved_data = pd.read_csv(test_path, keep_default_na=False)
    assert saved_data.loc[0, "validated_species"] == ""
    assert saved_data.loc[0, "predicted_species"] == "Gentoo"


@pytest.mark.parametrize("validated_species", ALLOWED_SPECIES)
def test_confirmed_species_is_saved_separately_from_prediction(
    tmp_path, observation, probabilities, validated_species
):
    """Die fachlich bestätigte Art darf von der Modellvorhersage abweichen."""
    test_path = tmp_path / "new_observations.csv"

    save_observation(
        observation,
        "Biscoe",
        "Gentoo",
        probabilities,
        test_path,
        validated_species=validated_species,
    )

    saved_data = pd.read_csv(test_path)
    assert list(saved_data.columns) == CSV_COLUMNS
    assert saved_data.loc[0, "validated_species"] == validated_species
    assert saved_data.loc[0, "predicted_species"] == "Gentoo"
    assert saved_data.loc[0, "probability_adelie"] == 0.002
    assert saved_data.loc[0, "probability_chinstrap"] == 0.0
    assert saved_data.loc[0, "probability_gentoo"] == 0.998


@pytest.mark.parametrize(
    "validated_species", ["Unknown", "gentoo", " Adelie ", "Nicht bestätigt", 123, False]
)
@pytest.mark.parametrize("file_exists", [False, True])
def test_invalid_confirmed_species_does_not_change_files(
    tmp_path, observation, probabilities, validated_species, file_exists
):
    """Ungültige Bestätigungen dürfen weder Dateien anlegen noch verändern."""
    test_path = tmp_path / "observations" / "new_observations.csv"
    if file_exists:
        save_observation(observation, "Biscoe", "Gentoo", probabilities, test_path)
        original_content = test_path.read_bytes()

    with pytest.raises(ValueError, match="bestätigte Art"):
        save_observation(
            observation,
            "Biscoe",
            "Gentoo",
            probabilities,
            test_path,
            validated_species=validated_species,
        )

    if file_exists:
        assert test_path.read_bytes() == original_content
    else:
        assert not test_path.exists()
        assert not test_path.parent.exists()


def test_appending_confirmed_and_unconfirmed_observations_preserves_existing_row(
    tmp_path, observation, probabilities
):
    """Alte Beobachtungen bleiben beim Anhängen mit und ohne Bestätigung erhalten."""
    test_path = tmp_path / "new_observations.csv"
    save_observation(observation, "Biscoe", "Gentoo", probabilities, test_path)
    original_content = test_path.read_bytes()
    original_row = pd.read_csv(test_path, keep_default_na=False).iloc[0]

    for validated_species in ("Adelie", None):
        save_observation(
            observation,
            "Biscoe",
            "Gentoo",
            probabilities,
            test_path,
            validated_species=validated_species,
        )

    saved_data = pd.read_csv(test_path, keep_default_na=False)
    assert test_path.read_bytes().startswith(original_content)
    assert len(saved_data) == 3
    pd.testing.assert_series_equal(saved_data.iloc[0], original_row)
    assert saved_data["validated_species"].tolist() == ["", "Adelie", ""]
