"""Tests für das Laden und Bereinigen der Palmer-Penguins-Daten."""

import pandas as pd

from src.data_processing import (
    NUMERIC_FEATURES,
    clean_data,
    load_clean_data,
)


def test_load_clean_data_has_expected_structure():
    """Der reale Datensatz soll nach der Bereinigung 342 Zeilen enthalten."""

    data = load_clean_data()

    assert len(data) == 342

    # Für die vier Körpermerkmale dürfen keine fehlenden Werte übrig bleiben.
    assert data[NUMERIC_FEATURES].isna().sum().sum() == 0

    # Fehlendes Geschlecht wird bewusst nicht zum Zeilenausschluss verwendet.
    assert data["sex"].isna().sum() == 9

    # Alle drei Zielklassen müssen weiterhin vorhanden sein.
    assert set(data["species"].unique()) == {
        "Adelie",
        "Chinstrap",
        "Gentoo",
    }


def test_clean_data_removes_missing_measurements_but_keeps_missing_sex():
    """Fehlende Körpermaße entfernen eine Zeile, fehlendes Geschlecht nicht."""

    test_data = pd.DataFrame(
        {
            "species": ["Adelie", "Gentoo", "Chinstrap"],
            "bill_length_mm": [39.1, 47.5, None],
            "bill_depth_mm": [18.7, 14.5, 18.2],
            "flipper_length_mm": [181, 218, 195],
            "body_mass_g": [3750, 5100, 3900],
            "sex": ["male", None, "female"],
            "island": ["Torgersen", "Biscoe", "Dream"],
        }
    )

    cleaned = clean_data(test_data)

    # Die dritte Zeile fehlt wegen bill_length_mm.
    assert len(cleaned) == 2

    # Der Gentoo-Pinguin mit unbekanntem Geschlecht bleibt erhalten.
    assert cleaned["sex"].isna().sum() == 1
    assert "Gentoo" in cleaned["species"].values
