"""Explorative Prüfung des Palmer-Penguins-Datensatzes."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "penguins.csv"

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]


def main() -> None:
    """Datensatz laden und zentrale Eigenschaften ausgeben."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Der Datensatz wurde nicht gefunden:\n{DATA_PATH}"
        )

    penguins = pd.read_csv(DATA_PATH)

    print("\n--- Grundübersicht ---")
    print(f"Datei: {DATA_PATH}")
    print(f"Anzahl Zeilen: {len(penguins)}")
    print(f"Anzahl Spalten: {len(penguins.columns)}")

    print("\n--- Spaltennamen ---")
    print(penguins.columns.tolist())

    print("\n--- Erste fünf Zeilen ---")
    print(penguins.head())

    print("\n--- Datentypen ---")
    print(penguins.dtypes)

    print("\n--- Fehlende Werte je Spalte ---")
    print(penguins.isna().sum())

    print("\n--- Anzahl der Pinguine je Art ---")
    print(penguins["species"].value_counts(dropna=False))

    print("\n--- Anzahl der Pinguine je Insel ---")
    print(penguins["island"].value_counts(dropna=False))

    print("\n--- Verteilung der Arten nach Insel ---")
    print(
        pd.crosstab(
            penguins["species"],
            penguins["island"],
            margins=True,
        )
    )

    print("\n--- Geschlechterverteilung nach Art ---")
    sex_for_analysis = penguins["sex"].fillna("missing")
    print(
        pd.crosstab(
            penguins["species"],
            sex_for_analysis,
            margins=True,
        )
    )

    print("\n--- Statistische Beschreibung der Messwerte ---")
    print(
        penguins[NUMERIC_FEATURES]
        .describe()
        .transpose()
        .round(2)
    )

    print("\n--- Wertebereiche nach Art ---")
    print(
        penguins
        .groupby("species")[NUMERIC_FEATURES]
        .agg(["min", "median", "max"])
        .round(2)
    )

    print("\n--- Vollständig doppelte Zeilen ---")
    print(penguins.duplicated().sum())

    print("\n--- Zeilen mit mindestens einem fehlenden Wert ---")
    missing_rows = penguins[penguins.isna().any(axis=1)]
    print(missing_rows.to_string(index=False))

    measurement_missing = penguins[NUMERIC_FEATURES].isna().any(axis=1)
    sex_missing = penguins["sex"].isna()

    print("\n--- Kombination fehlender Messwerte und Geschlechtsangaben ---")
    print(
        pd.crosstab(
            measurement_missing,
            sex_missing,
            rownames=["Mindestens ein Messwert fehlt"],
            colnames=["Geschlecht fehlt"],
            margins=True,
        )
    )


if __name__ == "__main__":
    main()