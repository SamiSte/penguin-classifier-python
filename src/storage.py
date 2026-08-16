"""Persistente Speicherung neuer Pinguinbeobachtungen als CSV-Datei."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PATH = PROJECT_ROOT / "data" / "new_observations.csv"


CSV_COLUMNS = [
    "timestamp_utc",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
    "predicted_species",
    "probability_adelie",
    "probability_chinstrap",
    "probability_gentoo",
    "validated_species",
]


def save_observation(
    observation: dict,
    island: str,
    predicted_species: str,
    probabilities: dict,
    output_path: Path = OBSERVATIONS_PATH,
) -> Path:
    """Speichere eine klassifizierte Beobachtung in einer CSV-Datei.

    Eine bestehende Datei wird nicht überschrieben. Neue Beobachtungen
    werden zeilenweise angehängt.

    Parameters
    ----------
    observation:
        Validierte Modellmerkmale der neuen Beobachtung.
    island:
        Erfasster Fundort. Dieser dient nur als Kontextinformation.
    predicted_species:
        Vom Modell vorhergesagte Art.
    probabilities:
        Klassenwahrscheinlichkeiten des Modells.
    output_path:
        Zielpfad der CSV-Datei.

    Returns
    -------
    pathlib.Path
        Pfad der verwendeten CSV-Datei.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "island": island,
        "bill_length_mm": observation["bill_length_mm"],
        "bill_depth_mm": observation["bill_depth_mm"],
        "flipper_length_mm": observation["flipper_length_mm"],
        "body_mass_g": observation["body_mass_g"],
        "sex": observation["sex"],
        "predicted_species": predicted_species,
        "probability_adelie": probabilities.get("Adelie"),
        "probability_chinstrap": probabilities.get("Chinstrap"),
        "probability_gentoo": probabilities.get("Gentoo"),
        # Darf später nach fachlicher Prüfung ergänzt werden.
        "validated_species": None,
    }

    new_row = pd.DataFrame(
        [row],
        columns=CSV_COLUMNS,
    )

    file_exists = output_path.exists()

    new_row.to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8",
    )

    return output_path