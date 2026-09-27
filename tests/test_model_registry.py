"""Modellwechsel dürfen weder Altmodelle verlieren noch halbe Versionen verwenden."""

from copy import deepcopy
import json

import joblib
import pandas as pd
import pytest

from src import model_registry as registry
from src import model_service
from src.data_processing import FEATURES, NUMERIC_FEATURES
from src.modeling import build_model_pipeline


@pytest.fixture
def registry_data(tmp_path):
    rows = []
    for species, length, depth, flipper, mass in (
        ("Adelie", 37.0, 18.0, 185.0, 3500.0),
        ("Chinstrap", 50.0, 19.0, 197.0, 3800.0),
        ("Gentoo", 48.0, 14.0, 220.0, 5100.0),
    ):
        for offset in range(3):
            rows.append({
                "bill_length_mm": length + offset,
                "bill_depth_mm": depth,
                "flipper_length_mm": flipper + offset,
                "body_mass_g": mass + 50 * offset,
                "sex": ("female", "male", "unknown")[offset],
                "species": species,
                "source": "reference",
                "feature_id": f"feature-{species}-{offset}",
                "record_id": f"row-{species}-{offset}",
            })
    data = pd.DataFrame(rows)
    model = build_model_pipeline()
    model.set_params(classifier__n_estimators=4, classifier__n_jobs=1)
    model.fit(data[FEATURES], data["species"])
    metadata = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": ["sex"],
        "classes": sorted(registry.SPECIES),
        "feature_ranges": {
            feature: {
                "minimum": float(data[feature].min()),
                "median": float(data[feature].median()),
                "maximum": float(data[feature].max()),
            } for feature in NUMERIC_FEATURES
        },
        "allowed_values": {"sex": sorted(registry.SEX_VALUES)},
    }
    metrics = {"test_metrics": {"accuracy": 1.0, "macro_f1": 1.0}}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "penguin_pipeline.joblib")
    (models_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (models_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return models_dir, model, metadata, metrics, data


def save_candidate(fixture, base="initial", metadata=None, metrics=None):
    models_dir, model, original_metadata, original_metrics, data = fixture
    return registry.save_candidate(
        model,
        metadata if metadata is not None else original_metadata,
        metrics if metrics is not None else original_metrics,
        data,
        {"base_version": base, "reference_hash": "a" * 64, "confirmed_ids": ["obs-1"]},
        models_dir,
    )


def test_default_registry_is_read_only(tmp_path):
    models_dir = tmp_path / "not-created"
    assert registry.get_registry_state(models_dir) == {"active_version": "initial", "previous_version": None}
    assert registry.get_manifest("initial", models_dir)["confirmed_ids"] == []
    assert not models_dir.exists()


def test_candidate_is_complete_but_not_active_and_retains_snapshot_columns(registry_data):
    models_dir, _, _, _, original_data = registry_data
    original_bytes = (models_dir / "penguin_pipeline.joblib").read_bytes()
    version = save_candidate(registry_data)
    paths = registry.version_paths(version, models_dir)
    assert all(path.is_file() for path in paths.values())
    assert list(pd.read_csv(paths["training_data"]).columns) == list(original_data.columns)
    assert registry.get_manifest(version, models_dir)["confirmed_ids"] == ["obs-1"]
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"
    assert not (models_dir / "active_model.json").exists()
    assert (models_dir / "penguin_pipeline.joblib").read_bytes() == original_bytes


def test_adoption_preserves_current_legacy_files_and_rollback(registry_data):
    models_dir = registry_data[0]
    metadata_path = models_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["user_note"] = "Eigene Anpassung muss erhalten bleiben"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    before = {key: (models_dir / registry.ARTIFACT_NAMES[key]).read_bytes() for key in ("model", "metadata", "metrics")}
    version = save_candidate(registry_data)
    state = registry.activate_candidate(version, models_dir)
    assert state == {"active_version": version, "previous_version": "initial"}
    initial = registry.version_paths("initial", models_dir)
    for key, content in before.items():
        assert initial[key].read_bytes() == content
        assert (models_dir / registry.ARTIFACT_NAMES[key]).read_bytes() == content
    assert registry.rollback(models_dir) == {"active_version": "initial", "previous_version": version}
    assert registry.version_paths(None, models_dir)["model"] == initial["model"]
    assert registry.get_manifest("initial", models_dir)["reference_hash"] == "a" * 64
    assert registry.rollback(models_dir)["active_version"] == version


def test_later_adoption_does_not_overwrite_initial_manifest(registry_data):
    models_dir, model, metadata, metrics, data = registry_data
    first = save_candidate(registry_data)
    registry.activate_candidate(first, models_dir)
    registry.rollback(models_dir)
    before = registry.version_paths("initial", models_dir)["manifest"].read_bytes()
    second = registry.save_candidate(
        model, metadata, metrics, data,
        {"base_version": "initial", "reference_hash": "b" * 64, "confirmed_ids": ["obs-2"]},
        models_dir,
    )
    registry.activate_candidate(second, models_dir)
    assert registry.version_paths("initial", models_dir)["manifest"].read_bytes() == before
    assert registry.get_manifest("initial", models_dir)["reference_hash"] == "a" * 64


def test_stale_candidate_cannot_replace_newer_active_version(registry_data):
    models_dir = registry_data[0]
    first = save_candidate(registry_data)
    stale = save_candidate(registry_data)
    registry.activate_candidate(first, models_dir)
    pointer_before = (models_dir / "active_model.json").read_bytes()
    with pytest.raises(ValueError, match="geändert"):
        registry.activate_candidate(stale, models_dir)
    assert (models_dir / "active_model.json").read_bytes() == pointer_before
    fresh = save_candidate(registry_data, base=first)
    assert registry.activate_candidate(fresh, models_dir)["previous_version"] == first


@pytest.mark.parametrize("version", ["../escape", "..", "a/b", "a\\b", "C:evil", "", None, "NUL", "COM1"])
def test_unsafe_version_ids_rejected_without_writes(tmp_path, version):
    with pytest.raises(ValueError, match="Ungültige Modellversion"):
        registry.activate_candidate(version, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("artifact", ["model", "metadata", "metrics", "training_data", "manifest"])
def test_incomplete_candidate_never_activates(registry_data, artifact):
    models_dir = registry_data[0]
    version = save_candidate(registry_data)
    registry.version_paths(version, models_dir)[artifact].unlink()
    with pytest.raises(ValueError, match="unvollständig"):
        registry.activate_candidate(version, models_dir)
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"
    assert not (models_dir / "versions" / "initial").exists()


def test_inconsistent_metadata_never_activates(registry_data):
    models_dir = registry_data[0]
    version = save_candidate(registry_data)
    path = registry.version_paths(version, models_dir)["metadata"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["classes"] = ["Adelie"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="drei Pinguinarten"):
        registry.activate_candidate(version, models_dir)
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"


def test_failed_candidate_save_leaves_no_half_version(registry_data):
    models_dir = registry_data[0]
    with pytest.raises(ValueError):
        save_candidate(registry_data, metrics={"accuracy": float("nan")})
    assert not list((models_dir / "versions").iterdir())
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"


def test_failed_atomic_pointer_update_keeps_previous_pointer(registry_data, monkeypatch):
    models_dir = registry_data[0]
    first = save_candidate(registry_data)
    registry.activate_candidate(first, models_dir)
    newer = save_candidate(registry_data, base=first)
    before = (models_dir / "active_model.json").read_bytes()

    def fail_replace(*args):
        raise OSError("Datenträger nicht beschreibbar")

    monkeypatch.setattr(registry.os, "replace", fail_replace)
    with pytest.raises(OSError):
        registry.activate_candidate(newer, models_dir)
    assert (models_dir / "active_model.json").read_bytes() == before
    assert not list(models_dir.glob(".active-*.json"))


def test_rollback_refuses_a_damaged_previous_model(registry_data):
    models_dir = registry_data[0]
    version = save_candidate(registry_data)
    registry.activate_candidate(version, models_dir)
    registry.version_paths("initial", models_dir)["model"].unlink()
    with pytest.raises(ValueError, match="unvollständig"):
        registry.rollback(models_dir)
    assert registry.get_registry_state(models_dir)["active_version"] == version


def test_model_service_reads_new_bundle_and_returns_version_without_restart(registry_data, monkeypatch):
    models_dir, _, metadata, _, data = registry_data
    monkeypatch.setattr(model_service, "MODELS_DIR", models_dir)
    observation = data.iloc[0][FEATURES].to_dict()
    observation["bill_length_mm"] = 70.0
    original = model_service.predict_species(observation)
    assert original["model_version"] == "initial"
    assert original["warnings"]
    new_metadata = deepcopy(metadata)
    new_metadata["feature_ranges"]["bill_length_mm"]["maximum"] = 80.0
    version = save_candidate(registry_data, metadata=new_metadata)
    registry.activate_candidate(version, models_dir)
    updated = model_service.predict_species(observation)
    assert updated["model_version"] == version
    assert updated["warnings"] == []
    registry.rollback(models_dir)
    assert model_service.predict_species(observation)["warnings"] == original["warnings"]


def test_prediction_keeps_its_bundle_if_model_changes_during_request(registry_data, monkeypatch):
    models_dir, _, metadata, _, data = registry_data
    monkeypatch.setattr(model_service, "MODELS_DIR", models_dir)
    new_metadata = deepcopy(metadata)
    new_metadata["feature_ranges"]["bill_length_mm"]["maximum"] = 80.0
    version = save_candidate(registry_data, metadata=new_metadata)
    original_loader = model_service.load_model_bundle
    load_count = 0

    def load_and_activate():
        nonlocal load_count
        load_count += 1
        bundle = original_loader()
        registry.activate_candidate(version, models_dir)
        return bundle

    monkeypatch.setattr(model_service, "load_model_bundle", load_and_activate)
    observation = data.iloc[0][FEATURES].to_dict()
    observation["bill_length_mm"] = 70.0
    result = model_service.predict_species(observation)
    assert result["model_version"] == "initial"
    assert result["warnings"]
    assert load_count == 1
    assert registry.get_registry_state(models_dir)["active_version"] == version


def test_initial_metadata_cli_changes_refresh_cache(registry_data, monkeypatch):
    models_dir = registry_data[0]
    monkeypatch.setattr(model_service, "MODELS_DIR", models_dir)
    before = model_service.load_metadata()
    changed = deepcopy(before)
    changed["cli_change"] = "neue Metadaten aus einem erneuten Trainingslauf"
    (models_dir / "model_metadata.json").write_text(json.dumps(changed), encoding="utf-8")
    assert model_service.load_metadata()["cli_change"] == changed["cli_change"]
