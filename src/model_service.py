"""Laden, Validieren und Anwenden des trainierten Pinguinmodells."""

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.formatting import format_number
from src.model_registry import (
    MODELS_DIR,
    REGISTRY_LOCK,
    get_registry_state,
    version_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = MODELS_DIR / "penguin_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

NUMERIC_LABELS = {
    "bill_length_mm": "Schnabellänge",
    "bill_depth_mm": "Schnabeltiefe",
    "flipper_length_mm": "Flossenlänge",
    "body_mass_g": "Körpergewicht",
}


@lru_cache(maxsize=4)
def _read_bundle(model_path: str, metadata_path: str, model_signature: tuple, metadata_signature: tuple):
    """Die Signaturen erneuern auch den Cache eines per CLI ersetzten Altmodells."""

    model = joblib.load(model_path)
    with Path(metadata_path).open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    return model, metadata


def load_model_bundle() -> tuple[Any, dict, str]:
    """Lade Modell, Metadaten und Versionskennung aus einem gemeinsamen Zustand.

    Der Registry-Lock verhindert, dass eine gleichzeitige Übernahme zwischen
    dem Lesen des Zeigers und der Auswahl der beiden Artefakte liegt.
    """

    with REGISTRY_LOCK:
        version = get_registry_state(MODELS_DIR)["active_version"]
        paths = version_paths(version, MODELS_DIR)
        signatures = []
        for key in ("model", "metadata"):
            if not paths[key].is_file():
                raise FileNotFoundError(
                    "Das Modell oder seine Metadaten wurden nicht gefunden:\n"
                    f"{paths[key]}\nFühre zuerst scripts/train_model.py aus."
                )
            stat = paths[key].stat()
            signatures.append((stat.st_mtime_ns, stat.st_size))
        model, metadata = _read_bundle(
            str(paths["model"]), str(paths["metadata"]), *signatures
        )
        return model, metadata, version


def load_model() -> Any:
    """Lade das aktuell übernommene Modell, ohne einen App-Neustart zu benötigen."""

    return load_model_bundle()[0]


def load_metadata() -> dict:
    """Lade die Metadaten der aktuell übernommenen Modellversion."""

    return load_model_bundle()[1]


def validate_observation(observation: dict, metadata: dict | None = None) -> tuple[dict, list[str]]:
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

    if metadata is None:
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
                f"{label}: Der Wert {format_number(numeric_value)} liegt außerhalb "
                f"des Trainingsbereichs von {format_number(minimum)} bis {format_number(maximum)}."
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

    model, metadata, model_version = load_model_bundle()
    cleaned_observation, warnings = validate_observation(observation, metadata)

    input_data = pd.DataFrame(
        [cleaned_observation],
        columns=metadata["features"],
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
        "model_version": model_version,
    }
