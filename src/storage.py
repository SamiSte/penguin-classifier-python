"""Persistente Speicherung neuer Pinguinbeobachtungen als CSV-Datei."""

from datetime import datetime, timezone
from pathlib import Path
import os
from threading import RLock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PATH = Path(os.environ.get("PENGUIN_OBSERVATIONS_PATH", PROJECT_ROOT / "data" / "new_observations.csv"))
OBSERVATIONS_LOCK = RLock()
ALLOWED_SPECIES = ("Adelie", "Chinstrap", "Gentoo")


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
    *,
    validated_species: str | None = None,
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
    validated_species:
        Unabhängig bestätigte tatsächliche Art. Ohne fachliche Bestätigung
        bleibt das Feld leer; die Modellvorhersage wird nicht übernommen.

    Returns
    -------
    pathlib.Path
        Pfad der verwendeten CSV-Datei.
    """

    if validated_species is not None and (
        not isinstance(validated_species, str)
        or validated_species not in ("", *ALLOWED_SPECIES)
    ):
        raise ValueError(
            "Die bestätigte Art muss Adelie, Chinstrap oder Gentoo sein. "
            "Ohne fachliche Bestätigung muss das Feld leer bleiben."
        )

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
        "validated_species": validated_species or None,
    }

    new_row = pd.DataFrame(
        [row],
        columns=CSV_COLUMNS,
    )

    with OBSERVATIONS_LOCK:
        file_exists = output_path.exists()
        new_row.to_csv(
            output_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
        )

    return output_path
