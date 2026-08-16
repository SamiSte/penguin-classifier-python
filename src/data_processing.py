"""Laden und Aufbereiten des Palmer-Penguins-Datensatzes."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "penguins.csv"

TARGET = "species"

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FEATURES = [
    "sex",
]

CONTEXT_FEATURES = [
    "island",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Lade den unveränderten Pinguin-Datensatz.

    Parameters
    ----------
    data_path:
        Pfad zur CSV-Datei.

    Returns
    -------
    pandas.DataFrame
        Unveränderter Datensatz.

    Raises
    ------
    FileNotFoundError
        Falls die CSV-Datei nicht vorhanden ist.
    ValueError
        Falls erforderliche Spalten fehlen.
    """

    if not data_path.exists():
        raise FileNotFoundError(
            f"Der Pinguin-Datensatz wurde nicht gefunden:\n{data_path}"
        )

    data = pd.read_csv(data_path)

    required_columns = {
    TARGET,
    *FEATURES,
    *CONTEXT_FEATURES,
    }
    
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Im Datensatz fehlen erforderliche Spalten: "
            f"{sorted(missing_columns)}"
        )

    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Bereite die Daten für Training und Modellbewertung auf.

    Entfernt werden nur Beobachtungen, bei denen mindestens einer der vier
    verpflichtenden Körpermesswerte fehlt. Ein fehlendes Geschlecht bleibt
    zunächst erhalten und wird später innerhalb der Modellpipeline durch die
    Kategorie ``unknown`` ersetzt.

    Das Erhebungsjahr wird nicht übernommen, da es kein fachlich sinnvolles
    Merkmal zur Bestimmung der Pinguinart darstellt.
    """

    selected_columns = [
    TARGET,
    *FEATURES,
    *CONTEXT_FEATURES,
    ]

    cleaned_data = (
        data.loc[:, selected_columns]
        .dropna(subset=NUMERIC_FEATURES)
        .reset_index(drop=True)
        .copy()
    )

    return cleaned_data


def load_clean_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Lade den Datensatz und führe die definierte Bereinigung durch."""

    raw_data = load_raw_data(data_path)
    return clean_data(raw_data)
