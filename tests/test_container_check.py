"""Lokale Prüfung der Docker-Probe; kein Ersatz für den echten Containerlauf."""

import json
from pathlib import Path
import shutil

import pytest

from scripts import container_check as probe
from src import model_service, retraining_data


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    shutil.copyfile(root / "data" / "penguins.csv", tmp_path / "data" / "penguins.csv")
    for name in ("penguin_pipeline.joblib", "model_metadata.json", "metrics.json"):
        shutil.copyfile(root / "models" / name, tmp_path / "models" / name)
    monkeypatch.setattr(model_service, "MODELS_DIR", tmp_path / "models")
    builder = retraining_data.build_model_pipeline
    monkeypatch.setattr(retraining_data, "build_model_pipeline",
                        lambda: builder().set_params(classifier__n_estimators=3,
                                                     classifier__n_jobs=1))
    yield tmp_path
    model_service._read_bundle.cache_clear()


def test_probe_checks_storage_training_persistence_and_rollback(workspace):
    probe.prepare(workspace, "penguin-pruefung-123-456-789")
    saved = json.loads((workspace / "data" / probe.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["candidate_id"] != "initial"
    assert "data/new_observations.csv" in saved["hashes"]
    assert "models/active_model.json" in saved["hashes"]
    # Ein neuer Prozess hat ebenfalls keinen bisherigen Inferenzcache.
    model_service._read_bundle.cache_clear()
    probe.verify(workspace, "penguin-pruefung-123-456-789")
    state = probe.registry.get_registry_state(workspace / "models")
    assert state["active_version"] == "initial"
    assert state["previous_version"] == saved["candidate_id"]


def test_probe_refuses_existing_observations_without_modifying_any_file(workspace):
    (workspace / "data" / "new_observations.csv").write_text("existing-data", encoding="utf-8")
    before = probe.file_hashes(workspace)
    with pytest.raises(RuntimeError, match="Vorhandene Daten"):
        probe.prepare(workspace, "penguin-pruefung-test123")
    assert probe.file_hashes(workspace) == before


def test_verify_detects_data_loss_before_attempting_rollback(workspace):
    probe.prepare(workspace, "penguin-pruefung-test123")
    pointer = workspace / "models" / "active_model.json"
    before = pointer.read_bytes()
    (workspace / "data" / "new_observations.csv").write_text("changed-data", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Dateien fehlen"):
        probe.verify(workspace, "penguin-pruefung-test123")
    assert pointer.read_bytes() == before


def test_verify_refuses_different_test_identity_without_changing_files(workspace):
    (workspace / "data" / probe.STATE_FILE).write_text(
        json.dumps({"test_id": "penguin-pruefung-other"}), encoding="utf-8")
    before = probe.file_hashes(workspace)
    with pytest.raises(RuntimeError, match="anderen Lauf"):
        probe.verify(workspace, "penguin-pruefung-test123")
    assert probe.file_hashes(workspace) == before


def test_probe_refuses_production_model_path(workspace, monkeypatch):
    monkeypatch.setattr(model_service, "MODELS_DIR", workspace / "different-models")
    before = probe.file_hashes(workspace)
    with pytest.raises(RuntimeError, match="Modellpfad"):
        probe.prepare(workspace, "penguin-pruefung-test123")
    assert probe.file_hashes(workspace) == before
