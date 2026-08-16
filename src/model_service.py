"""Laden, Validieren und Anwenden des trainierten Pinguinmodells."""

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "penguin_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

NUMERIC_LABELS = {
    "bill_length_mm": "Schnabellänge",
    "bill_depth_mm": "Schnabeltiefe",
    "flipper_length_mm": "Flossenlänge",
    "body_mass_g": "Körpergewicht",
}


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Lade das trainierte Modell und behalte es im Arbeitsspeicher."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Das trainierte Modell wurde nicht gefunden:\n"
            f"{MODEL_PATH}\n"
            "Führe zuerst scripts/train_model.py aus."
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Lade die zum Modell gehörenden Metadaten."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Die Modellmetadaten wurden nicht gefunden:\n"
            f"{METADATA_PATH}\n"
            "Führe zuerst scripts/train_model.py aus."
        )

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def validate_observation(observation: dict) -> tuple[dict, list[str]]:
    """Prüfe eine neue Beobachtung und bereite sie für das Modell auf.

    Die vier Körpermesswerte sind verpflichtend. Das Geschlecht darf als
    ``unknown`` angegeben werden. Werte außerhalb des Trainingsbereichs werden
    nicht blockiert, aber als Warnung zurückgegeben.

    Parameters
    ----------
    observation:
        Wörterbuch mit den fünf Modellmerkmalen.

    Returns
    -------
    tuple[dict, list[str]]
        Bereinigte Beobachtung und mögliche Warnhinweise.

    Raises
    ------
    ValueError
        Bei fehlenden, ungültigen oder nicht endlichen Eingaben.
    """

    metadata = load_metadata()
    expected_features = metadata["features"]
    numeric_features = metadata["numeric_features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in observation
    ]

    if missing_features:
        raise ValueError(
            "Folgende Eingaben fehlen: "
            + ", ".join(missing_features)
        )

    cleaned_observation = {}
    warnings = []

    for feature in numeric_features:
        raw_value = observation.get(feature)

        if raw_value is None or raw_value == "":
            label = NUMERIC_LABELS.get(feature, feature)
            raise ValueError(
                f"Für „{label}“ muss ein Wert eingegeben werden."
            )

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            label = NUMERIC_LABELS.get(feature, feature)
            raise ValueError(
                f"Die Eingabe für „{label}“ ist keine gültige Zahl."
            ) from error

        if not math.isfinite(numeric_value):
            label = NUMERIC_LABELS.get(feature, feature)
            raise ValueError(
                f"Die Eingabe für „{label}“ muss eine endliche Zahl sein."
            )

        if numeric_value <= 0:
            label = NUMERIC_LABELS.get(feature, feature)
            raise ValueError(
                f"Die Eingabe für „{label}“ muss größer als null sein."
            )

        feature_range = metadata["feature_ranges"][feature]
        minimum = feature_range["minimum"]
        maximum = feature_range["maximum"]

        if numeric_value < minimum or numeric_value > maximum:
            label = NUMERIC_LABELS.get(feature, feature)
            warnings.append(
                f"{label}: Der Wert {numeric_value:g} liegt außerhalb "
                f"des Trainingsbereichs von {minimum:g} bis {maximum:g}."
            )

        cleaned_observation[feature] = numeric_value

    sex = observation.get("sex")

    if sex is None or sex == "":
        sex = "unknown"

    allowed_sex_values = metadata["allowed_values"]["sex"]

    if sex not in allowed_sex_values:
        raise ValueError(
            "Das Geschlecht muss female, male oder unknown sein."
        )

    cleaned_observation["sex"] = sex

    return cleaned_observation, warnings


def predict_species(observation: dict) -> dict:
    """Bestimme die Pinguinart und liefere Klassenwahrscheinlichkeiten."""

    cleaned_observation, warnings = validate_observation(observation)

    model = load_model()

    input_data = pd.DataFrame(
        [cleaned_observation],
        columns=load_metadata()["features"],
    )

    predicted_species = str(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    probability_by_species = {
        str(species): float(probability)
        for species, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return {
        "predicted_species": predicted_species,
        "probabilities": probability_by_species,
        "warnings": warnings,
        "validated_observation": cleaned_observation,
    }