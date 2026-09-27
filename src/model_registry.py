"""Lokale Modellversionen mit ausdrücklicher Übernahme und Rückkehrmöglichkeit.

Jede Version besitzt einen unveränderlichen Ordner. Nur ``active_model.json``
wird atomar ersetzt. Das Prozess-Lock schützt den lokalen Dash-Einzelbetrieb;
mehrere Serverprozesse benötigen eine gemeinsame, externe Koordination.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from threading import RLock
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.data_processing import FEATURES, NUMERIC_FEATURES, TARGET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("PENGUIN_MODEL_DIR", PROJECT_ROOT / "models"))
REGISTRY_LOCK = RLock()
ARTIFACT_NAMES = {
    "model": "penguin_pipeline.joblib",
    "metadata": "model_metadata.json",
    "metrics": "metrics.json",
    "training_data": "training_data.csv",
    "manifest": "manifest.json",
}
SPECIES = {"Adelie", "Chinstrap", "Gentoo"}
SEX_VALUES = {"female", "male", "unknown"}


def _validate_version_id(version_id: str) -> str:
    """Erlaube ausschließlich einfache Ordnernamen innerhalb der Registry."""

    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if (
        not isinstance(version_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", version_id)
        or version_id.upper() in reserved
    ):
        raise ValueError("Ungültige Modellversion.")
    return version_id


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Ungültige JSON-Struktur in {path.name}.")
    return result


def _write_json(path: Path, content: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def get_registry_state(models_dir=MODELS_DIR) -> dict:
    """Lies den aktiven Zeiger; ein vorhandenes Altmodell bleibt unverändert."""

    with REGISTRY_LOCK:
        pointer = Path(models_dir) / "active_model.json"
        if not pointer.exists():
            return {"active_version": "initial", "previous_version": None}
        state = _read_json(pointer)
        active = _validate_version_id(state.get("active_version"))
        previous = state.get("previous_version")
        if previous is not None:
            _validate_version_id(previous)
        if previous == active:
            raise ValueError("Aktive und vorherige Modellversion dürfen nicht identisch sein.")
        return {"active_version": active, "previous_version": previous}


def version_paths(version_id=None, models_dir=MODELS_DIR) -> dict:
    """Ermittle zusammengehörende Artefakte einer expliziten oder aktiven Version."""

    with REGISTRY_LOCK:
        root = Path(models_dir)
        if version_id is None:
            version_id = get_registry_state(root)["active_version"]
        _validate_version_id(version_id)
        directory = root / "versions" / version_id
        if version_id == "initial" and not directory.exists():
            directory = root
        return {name: directory / filename for name, filename in ARTIFACT_NAMES.items()}


def get_manifest(version_id, models_dir=MODELS_DIR) -> dict:
    """Das ursprüngliche Modell hat noch keine bestätigten Zusatzbeobachtungen."""

    paths = version_paths(version_id, models_dir)
    if version_id == "initial" and not paths["manifest"].exists():
        return {"version_id": "initial", "confirmed_ids": []}
    manifest = _read_json(paths["manifest"])
    if manifest.get("version_id") != version_id:
        raise ValueError("Manifest und Modellversion stimmen nicht überein.")
    return manifest


def validate_model_bundle(model, metadata: dict) -> None:
    """Prüfe das Vorhersageschema und einen Probelauf vor einem Modellwechsel."""

    try:
        if metadata["features"] != FEATURES or metadata["numeric_features"] != NUMERIC_FEATURES:
            raise ValueError("Die Modellmerkmale stimmen nicht mit der Anwendung überein.")
        if metadata["categorical_features"] != ["sex"]:
            raise ValueError("Die kategorialen Modellmerkmale sind ungültig.")
        classes = list(map(str, model.classes_))
        if set(metadata["classes"]) != SPECIES or set(classes) != SPECIES or len(classes) != 3:
            raise ValueError("Das Modell muss alle drei Pinguinarten unterscheiden.")
        if set(metadata["allowed_values"]["sex"]) != SEX_VALUES:
            raise ValueError("Die Geschlechtskategorien stimmen nicht überein.")
        model_features = getattr(model, "feature_names_in_", FEATURES)
        if list(model_features) != FEATURES:
            raise ValueError("Modell und Metadaten verwenden unterschiedliche Merkmale.")
        example = {}
        for feature in NUMERIC_FEATURES:
            limits = metadata["feature_ranges"][feature]
            low, median, high = (float(limits[key]) for key in ("minimum", "median", "maximum"))
            if not all(math.isfinite(value) for value in (low, median, high)) or not 0 < low <= median <= high:
                raise ValueError("Die Wertebereiche der Modellmetadaten sind ungültig.")
            example[feature] = median
        example["sex"] = "unknown"
        frame = pd.DataFrame([example], columns=FEATURES)
        prediction = model.predict(frame)
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        if len(prediction) != 1 or str(prediction[0]) not in SPECIES:
            raise ValueError("Das Modell liefert keine gültige Pinguinart.")
        if (
            probabilities.shape != (1, 3)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0)
        ):
            raise ValueError("Das Modell liefert ungültige Klassenwahrscheinlichkeiten.")
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Das Modell oder seine Metadaten sind unvollständig.") from error


def _validate_training_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame) or data.empty or not set(FEATURES + [TARGET]).issubset(data.columns):
        raise ValueError("Der Trainingssnapshot ist unvollständig.")
    try:
        numbers = data[NUMERIC_FEATURES].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Der Trainingssnapshot enthält ungültige Messwerte.") from error
    if not np.isfinite(numbers).all() or (numbers <= 0).any():
        raise ValueError("Der Trainingssnapshot enthält ungültige Messwerte.")
    if not data["sex"].isin(SEX_VALUES).all() or not data[TARGET].isin(SPECIES).all():
        raise ValueError("Der Trainingssnapshot enthält ungültige Kategorien.")
    if set(data[TARGET]) != SPECIES:
        raise ValueError("Im Trainingssnapshot fehlen Pinguinarten.")


def _validate_manifest(manifest: dict, version_id=None) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Das Trainingsmanifest fehlt.")
    _validate_version_id(manifest.get("base_version"))
    if not isinstance(manifest.get("reference_hash"), str) or not manifest["reference_hash"]:
        raise ValueError("Der Fingerabdruck der Referenzdaten fehlt.")
    confirmed_ids = manifest.get("confirmed_ids")
    if (
        not isinstance(confirmed_ids, list)
        or any(not isinstance(value, str) or not value for value in confirmed_ids)
        or len(set(confirmed_ids)) != len(confirmed_ids)
    ):
        raise ValueError("Die Kennungen der bestätigten Beobachtungen sind ungültig.")
    if version_id is not None and manifest.get("version_id") != version_id:
        raise ValueError("Manifest und Modellversion stimmen nicht überein.")


def _remove_staging(staging: Path, versions: Path) -> None:
    # Ausschließlich den selbst angelegten, noch unveröffentlichten Ordner entfernen.
    if staging.exists() and staging.resolve().parent == versions.resolve() and staging.name.startswith(".staging-"):
        shutil.rmtree(staging)


def save_candidate(model, metadata, metrics, training_data, manifest, models_dir=MODELS_DIR) -> str:
    """Speichere einen geprüften Kandidaten vollständig, ohne ihn zu aktivieren."""

    with REGISTRY_LOCK:
        validate_model_bundle(model, metadata)
        _validate_training_data(training_data)
        _validate_manifest(manifest)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Die Modellbewertung fehlt.")
        version_id = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:12]
        versions = Path(models_dir) / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        staging = versions / (".staging-" + uuid4().hex)
        staging.mkdir()
        saved_manifest = {
            **manifest,
            "version_id": version_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            joblib.dump(model, staging / ARTIFACT_NAMES["model"])
            _write_json(staging / ARTIFACT_NAMES["metadata"], metadata)
            _write_json(staging / ARTIFACT_NAMES["metrics"], metrics)
            training_data.to_csv(staging / ARTIFACT_NAMES["training_data"], index=False)
            _write_json(staging / ARTIFACT_NAMES["manifest"], saved_manifest)
            # rename veröffentlicht den vollständigen Ordner auf demselben Laufwerk.
            staging.rename(versions / version_id)
        finally:
            _remove_staging(staging, versions)
        return version_id


def _validate_version(version_id: str, models_dir) -> dict:
    paths = version_paths(version_id, models_dir)
    required = ("model", "metadata", "metrics") if version_id == "initial" else ARTIFACT_NAMES
    if any(not paths[key].is_file() for key in required):
        raise ValueError("Die Modellversion ist unvollständig und kann nicht verwendet werden.")
    model = joblib.load(paths["model"])
    metadata = _read_json(paths["metadata"])
    validate_model_bundle(model, metadata)
    _read_json(paths["metrics"])
    manifest = get_manifest(version_id, models_dir)
    if version_id != "initial":
        _validate_manifest(manifest, version_id)
        _validate_training_data(pd.read_csv(paths["training_data"], keep_default_na=False))
    return manifest


def _preserve_initial(models_dir, reference_hash: str) -> None:
    root = Path(models_dir)
    versions = root / "versions"
    initial = versions / "initial"
    if initial.exists():
        _validate_version("initial", root)
        return
    versions.mkdir(parents=True, exist_ok=True)
    staging = versions / (".staging-" + uuid4().hex)
    staging.mkdir()
    try:
        for key in ("model", "metadata", "metrics"):
            shutil.copy2(root / ARTIFACT_NAMES[key], staging / ARTIFACT_NAMES[key])
        _write_json(staging / ARTIFACT_NAMES["manifest"], {
            "version_id": "initial", "confirmed_ids": [], "reference_hash": reference_hash,
        })
        validate_model_bundle(joblib.load(staging / ARTIFACT_NAMES["model"]), _read_json(staging / ARTIFACT_NAMES["metadata"]))
        staging.rename(initial)
    finally:
        _remove_staging(staging, versions)


def _write_pointer(state: dict, models_dir) -> None:
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / (".active-" + uuid4().hex + ".json")
    try:
        _write_json(temporary, state)
        os.replace(temporary, root / "active_model.json")
    finally:
        temporary.unlink(missing_ok=True)


def activate_candidate(version_id, models_dir=MODELS_DIR) -> dict:
    """Aktiviere nur einen vollständigen Kandidaten zur weiterhin aktiven Basis."""

    with REGISTRY_LOCK:
        _validate_version_id(version_id)
        if version_id == "initial":
            raise ValueError("Für die vorherige Version bitte die Rückkehrfunktion verwenden.")
        state = get_registry_state(models_dir)
        manifest = _validate_version(version_id, models_dir)
        if manifest["base_version"] != state["active_version"]:
            raise ValueError("Das aktive Modell hat sich geändert. Bitte erneut trainieren und vergleichen.")
        if state["active_version"] == "initial":
            _validate_version("initial", models_dir)
            _preserve_initial(models_dir, manifest["reference_hash"])
        next_state = {"active_version": version_id, "previous_version": state["active_version"]}
        _write_pointer(next_state, models_dir)
        return next_state


def rollback(models_dir=MODELS_DIR) -> dict:
    """Wechsle zur vorherigen Version; beide Versionen bleiben vollständig erhalten."""

    with REGISTRY_LOCK:
        state = get_registry_state(models_dir)
        previous = state["previous_version"]
        if previous is None:
            raise ValueError("Es gibt noch keine vorherige Modellversion.")
        _validate_version(previous, models_dir)
        next_state = {"active_version": previous, "previous_version": state["active_version"]}
        _write_pointer(next_state, models_dir)
        return next_state
