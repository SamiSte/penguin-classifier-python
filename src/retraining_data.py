"""Geprüfte Trainingsdaten und vergleichbare Bewertungen für Re-Training.

Vorhersagen sind keine Zielwerte: Zusätzliche Beobachtungen werden ausschließlich
mit ihrer unabhängig bestätigten Art übernommen. Beide Modellvarianten werden
für den Vergleich frisch trainiert und sehen denselben Referenz-Prüfanteil nicht.
Das spätere Auslieferungsmodell wird anschließend auf allen Daten trainiert.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data_processing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_clean_data,
)
from src.modeling import RANDOM_STATE, build_model_pipeline
from src.storage import ALLOWED_SPECIES, OBSERVATIONS_LOCK


DATASET_COLUMNS = [*FEATURES, TARGET, "source", "feature_id", "record_id"]
SEX_VALUES = ["female", "male", "unknown"]
TEST_FRACTION = 0.25
CV_FOLDS = 5


def _hash(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_row(row: dict, label: str, source: str, location: str) -> dict:
    """Normalisiere Merkmale vor Identitätsprüfung und Training."""
    if label not in ALLOWED_SPECIES:
        raise ValueError(
            f"{location}: Die bestätigte Art muss Adelie, Chinstrap oder Gentoo sein."
        )
    cleaned = {}
    for feature in NUMERIC_FEATURES:
        try:
            value = float(row[feature])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"{location}: {feature} muss eine endliche Zahl größer als null sein."
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{location}: {feature} muss eine endliche Zahl größer als null sein."
            )
        cleaned[feature] = value
    sex = row.get("sex")
    if sex is None or pd.isna(sex) or str(sex).strip().lower() in (
        "", "na", "nan", "none", "null"
    ):
        sex = "unknown"
    else:
        sex = str(sex).strip().lower()
    if sex not in SEX_VALUES:
        raise ValueError(
            f"{location}: Das Geschlecht muss female, male oder unknown sein."
        )
    cleaned["sex"] = sex
    values = [cleaned[feature] for feature in FEATURES]
    cleaned.update(
        species=label,
        source=source,
        feature_id=_hash(values),
        record_id=_hash([*values, label]),
    )
    return cleaned


def _confirmed_rows(observations_path: Path):
    """Lese den vollständigen CSV-Stand atomar gegenüber lokalen Speicheraktionen."""
    with OBSERVATIONS_LOCK:
        return _read_observation_rows(observations_path)


def _read_observation_rows(observations_path: Path) -> list[tuple[int, dict]]:
    """Lese CSV strikt und liefere tatsächliche CSV-Zeilennummern mit."""
    if not observations_path.exists():
        return []
    rows = []
    try:
        with observations_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            required = {*FEATURES, "validated_species"}
            if not header or required.difference(header):
                missing = sorted(required.difference(header or []))
                raise ValueError(
                    "Beobachtungs-CSV, Zeile 1: Erforderliche Spalten fehlen: "
                    + ", ".join(missing)
                )
            if len(header) != len(set(header)):
                raise ValueError("Beobachtungs-CSV, Zeile 1: Spaltennamen sind doppelt.")
            for fields in reader:
                line_number = reader.line_num
                if not fields:
                    continue
                if len(fields) != len(header):
                    raise ValueError(
                        f"Beobachtungs-CSV, Zeile {line_number}: "
                        "Die Anzahl der Werte passt nicht zu den Spalten."
                    )
                rows.append((line_number, dict(zip(header, fields))))
    except csv.Error as error:
        raise ValueError(
            f"Beobachtungs-CSV, Zeile {reader.line_num}: CSV kann nicht gelesen werden."
        ) from error
    except (OSError, UnicodeError) as error:
        raise ValueError("Die Beobachtungs-CSV kann nicht als UTF-8 gelesen werden.") from error
    return rows


def prepare_dataset(reference_path: Path, observations_path: Path) -> tuple[pd.DataFrame, dict]:
    """Erzeuge einen geprüften, deduplizierten Snapshot ohne Dateien zu verändern."""
    try:
        reference = load_clean_data(Path(reference_path))
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Die Referenzdaten können nicht geladen werden: {error}") from error
    if reference.empty:
        raise ValueError("Die Referenzdaten enthalten keine vollständigen Beobachtungen.")

    rows = []
    seen = {}
    duplicates_removed = 0
    ignored_unconfirmed = 0

    def add_row(row: dict, label: str, source: str, location: str) -> None:
        nonlocal duplicates_removed
        cleaned = _normalise_row(row, label, source, location)
        existing = seen.get(cleaned["feature_id"])
        if existing is not None:
            previous, previous_location = existing
            if previous[TARGET] != cleaned[TARGET]:
                raise ValueError(
                    f"{location}: Widersprüchliche Arten für dieselben Merkmale "
                    f"({cleaned[TARGET]} und {previous[TARGET]}; {previous_location}). "
                    "Bitte die fachlichen Bestätigungen prüfen."
                )
            duplicates_removed += 1
            return
        seen[cleaned["feature_id"]] = (cleaned, location)
        rows.append(cleaned)

    for index, row in enumerate(reference.to_dict("records"), start=1):
        add_row(row, row[TARGET], "reference", f"Referenzbeobachtung {index}")
    reference_ids = sorted(row["record_id"] for row in rows)
    for line_number, row in _confirmed_rows(Path(observations_path)):
        label = row["validated_species"].strip()
        if not label:
            ignored_unconfirmed += 1
            continue
        add_row(row, label, "confirmed", f"Beobachtungs-CSV, Zeile {line_number}")

    data = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    confirmed = data.loc[data["source"].eq("confirmed")]
    warnings = []
    if duplicates_removed:
        warnings.append(
            f"{duplicates_removed} doppelte Beobachtung(en) wurden nur einmal berücksichtigt."
        )
    if len(confirmed) and confirmed[TARGET].nunique() < len(ALLOWED_SPECIES):
        warnings.append(
            "Die neuen bestätigten Beobachtungen decken nicht alle drei Arten ab. "
            "Die ursprünglichen Referenzdaten bleiben im Training enthalten; "
            "eine automatische Klassengewichtung erfolgt nicht."
        )
    report = {
        "confirmed_count": int(len(confirmed)),
        "ignored_unconfirmed": ignored_unconfirmed,
        "duplicates_removed": duplicates_removed,
        "class_counts": {name: int(data[TARGET].eq(name).sum()) for name in ALLOWED_SPECIES},
        "confirmed_class_counts": {
            name: int(confirmed[TARGET].eq(name).sum()) for name in ALLOWED_SPECIES
        },
        "reference_hash": _hash(reference_ids),
        "confirmed_ids": sorted(confirmed["record_id"].tolist()),
        "warnings": warnings,
    }
    return data, report


def _snapshot_reference(data: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = set(DATASET_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"{label}: Im Trainingssnapshot fehlen Spalten: {sorted(missing)}")
    if data.empty or not data["source"].isin(["reference", "confirmed"]).all():
        raise ValueError(f"{label}: Ungültige oder leere Datenherkunft im Trainingssnapshot.")
    if not data[TARGET].isin(ALLOWED_SPECIES).all():
        raise ValueError(f"{label}: Ungültige Art im Trainingssnapshot.")
    for index, row in enumerate(data.to_dict("records"), start=1):
        checked = _normalise_row(row, row[TARGET], row["source"], f"{label}, Zeile {index}")
        if any(row[key] != checked[key] for key in ("feature_id", "record_id")):
            raise ValueError(f"{label}, Zeile {index}: Die Trainingsdaten wurden verändert.")
    reference = data.loc[data["source"].eq("reference")].copy()
    if set(reference[TARGET]) != set(ALLOWED_SPECIES):
        raise ValueError(f"{label}: Die Referenzdaten müssen alle drei Arten enthalten.")
    return reference.sort_values("record_id").reset_index(drop=True)


def _scores(model, validation: pd.DataFrame) -> dict:
    predicted = model.predict(validation[FEATURES])
    actual = validation[TARGET]
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(actual, predicted)),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(ALLOWED_SPECIES)
        ).tolist(),
    }


def train_and_evaluate(
    candidate_data: pd.DataFrame,
    baseline_data: pd.DataFrame,
    progress: Callable[[str], None] | None = None,
    *,
    baseline_pipeline=None,
) -> dict:
    """Vergleiche zwei Datenstände auf demselben zurückgehaltenen Referenzanteil.

    Die aktive Pipeline wird bewusst nicht für die Bewertung wiederverwendet:
    Das Auslieferungsmodell hat auch die Referenz-Prüfdaten bereits gesehen.
    Wiederholte Vergleiche ersetzen keine unabhängige externe Validierung.
    """
    report_progress = progress or (lambda message: None)
    candidate_reference = _snapshot_reference(candidate_data, "Neue Version")
    baseline_reference = _snapshot_reference(baseline_data, "Bisherige Version")
    if candidate_reference["record_id"].tolist() != baseline_reference["record_id"].tolist():
        raise ValueError(
            "Die Referenzdaten unterscheiden sich von der aktiven Modellversion. "
            "Ein vergleichbarer Modelltest ist so nicht möglich."
        )
    try:
        _, validation = train_test_split(
            candidate_reference,
            test_size=TEST_FRACTION,
            random_state=RANDOM_STATE,
            stratify=candidate_reference[TARGET],
        )
    except ValueError as error:
        raise ValueError("Zu wenige Referenzdaten für einen stratifizierten Modellvergleich.") from error
    validation_ids = set(validation["feature_id"])
    candidate_train = candidate_data.loc[~candidate_data["feature_id"].isin(validation_ids)]
    baseline_train = baseline_data.loc[~baseline_data["feature_id"].isin(validation_ids)]
    counts = candidate_train[TARGET].value_counts()
    if any(counts.get(name, 0) < CV_FOLDS for name in ALLOWED_SPECIES):
        raise ValueError(
            "Für die fünffache Kreuzvalidierung werden mindestens fünf "
            "Trainingsbeobachtungen je Art benötigt."
        )

    report_progress("Bisherigen Datenstand auf dem festen Prüfanteil bewerten …")
    # Nur die Konfiguration übernehmen, niemals den bereits gelernten Zustand.
    baseline_model = (clone(baseline_pipeline) if baseline_pipeline is not None
                      else build_model_pipeline())
    baseline_model.fit(baseline_train[FEATURES], baseline_train[TARGET])
    baseline_metrics = _scores(baseline_model, validation)

    report_progress("Neue Modellvariante trainieren und auf demselben Prüfanteil bewerten …")
    candidate_model = build_model_pipeline()
    candidate_model.fit(candidate_train[FEATURES], candidate_train[TARGET])
    candidate_metrics = _scores(candidate_model, validation)

    report_progress("Fünffache Kreuzvalidierung auf den Trainingsdaten durchführen …")
    cv_results = cross_validate(
        build_model_pipeline(),
        candidate_train[FEATURES],
        candidate_train[TARGET],
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
            "cohen_kappa": make_scorer(cohen_kappa_score),
        },
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        error_score="raise",
    )
    cv_summary = {
        name: {
            "mean": float(np.mean(cv_results[f"test_{name}"])),
            "standard_deviation": float(np.std(cv_results[f"test_{name}"])),
            "individual_scores": [float(value) for value in cv_results[f"test_{name}"]],
        }
        for name in ("accuracy", "macro_f1", "cohen_kappa")
    }

    report_progress("Auslieferungsmodell auf allen geprüften Daten trainieren …")
    deployment_model = build_model_pipeline()
    deployment_model.fit(candidate_data[FEATURES], candidate_data[TARGET])
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_ranges": {
            feature: {
                "minimum": float(candidate_data[feature].min()),
                "maximum": float(candidate_data[feature].max()),
                "median": float(candidate_data[feature].median()),
            }
            for feature in NUMERIC_FEATURES
        },
        "allowed_values": {
            "sex": SEX_VALUES,
            "island": ["Biscoe", "Dream", "Torgersen"],
        },
        "classes": list(ALLOWED_SPECIES),
        "model_parameters": {
            key: deployment_model.named_steps["classifier"].get_params()[key]
            for key in ("n_estimators", "max_depth", "min_samples_leaf")
        },
    }
    metrics = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "cross_validation": cv_summary,
        "classes": list(ALLOWED_SPECIES),
        "validation_size": int(len(validation)),
        "baseline_training_size": int(len(baseline_train)),
        "candidate_training_size": int(len(candidate_train)),
        "validation_feature_ids": sorted(validation_ids),
        "test_fraction": TEST_FRACTION,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "baseline_parameters": {
            key: baseline_model.named_steps["classifier"].get_params()[key]
            for key in ("n_estimators", "max_depth", "min_samples_leaf")
        },
        "candidate_parameters": metadata["model_parameters"],
    }
    return {"model": deployment_model, "metadata": metadata, "metrics": metrics}
