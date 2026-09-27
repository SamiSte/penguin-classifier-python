"""Integration des manuellen Trainings mit isolierten Modellen und Beobachtungen."""

import json
from pathlib import Path
import shutil
from threading import Event

import pandas as pd
import pytest

from src import model_registry as registry
from src import retraining, retraining_data
from src.data_processing import DATA_PATH
from src.retraining import RetrainingService
from src.storage import save_observation


@pytest.fixture
def training_workspace(tmp_path, monkeypatch):
    """Originalartefakte werden nur gelesen; jede Änderung bleibt im Testordner."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    original_models = Path(__file__).resolve().parents[1] / "models"
    for key in ("model", "metadata", "metrics"):
        filename = registry.ARTIFACT_NAMES[key]
        shutil.copyfile(original_models / filename, models_dir / filename)
    reference = tmp_path / "reference.csv"
    shutil.copyfile(DATA_PATH, reference)
    observations = tmp_path / "observations.csv"
    builder = retraining_data.build_model_pipeline

    def small_pipeline():
        return builder().set_params(classifier__n_estimators=3, classifier__n_jobs=1)

    monkeypatch.setattr(retraining_data, "build_model_pipeline", small_pipeline)
    services = []

    def new_service():
        service = RetrainingService(models_dir, reference, observations)
        services.append(service)
        return service

    yield models_dir, reference, observations, new_service
    for service in services:
        service.close()


def add_observation(path, confirmed="Chinstrap", length=49.12345678901235):
    save_observation(
        observation={
            "bill_length_mm": length,
            "bill_depth_mm": 18.12345678901235,
            "flipper_length_mm": 201,
            "body_mass_g": 4123,
            "sex": "unknown",
        },
        island="Dream",
        predicted_species="Adelie",
        probabilities={"Adelie": 0.7, "Chinstrap": 0.2, "Gentoo": 0.1},
        output_path=path,
        validated_species=confirmed,
    )


def artifact_bytes(models_dir):
    return {
        key: (models_dir / registry.ARTIFACT_NAMES[key]).read_bytes()
        for key in ("model", "metadata", "metrics")
    }


def train_ready(service):
    service.start_training()
    service.wait(timeout=30)
    status = service.get_status()
    assert status["phase"] == "ready", status["message"]
    assert status["can_adopt"]
    return status


def test_initial_status_cannot_train_without_confirmed_data(training_workspace):
    models_dir, _, observations, create = training_workspace
    before = artifact_bytes(models_dir)
    service = create()
    status = service.get_status()
    assert status["active_version"] == "initial"
    assert status["phase"] == "idle"
    assert status["data_report"]["confirmed_count"] == 0
    assert status["new_confirmed_count"] == 0
    assert not status["can_train"]
    assert not status["can_adopt"]
    assert not status["can_rollback"]
    with pytest.raises(ValueError, match="Keine neuen"):
        service.start_training()
    assert not observations.exists()
    assert artifact_bytes(models_dir) == before
    assert not (models_dir / "active_model.json").exists()


def test_submit_failure_does_not_leave_training_stuck(training_workspace):
    _, _, observations, create = training_workspace
    add_observation(observations)
    service = create()
    service.close()
    with pytest.raises(ValueError, match="nicht gestartet"):
        service.start_training()
    assert service.get_status()["phase"] == "error"


def test_training_is_single_job_and_keeps_active_model_until_explicit_adoption(
    training_workspace, monkeypatch
):
    models_dir, _, observations, create = training_workspace
    before = artifact_bytes(models_dir)
    add_observation(observations)
    service = create()
    assert service.get_status()["new_confirmed_count"] == 1
    assert service.get_status()["can_train"]
    entered = Event()
    release = Event()
    evaluate = retraining.train_and_evaluate

    def blocked_evaluation(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("Testfreigabe fehlt")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(retraining, "train_and_evaluate", blocked_evaluation)
    service.start_training()
    try:
        assert entered.wait(10)
        busy = service.get_status()
        assert busy["phase"] == "training"
        assert not busy["can_train"]
        assert not busy["can_adopt"]
        with pytest.raises(ValueError, match="bereits"):
            service.start_training()
        with pytest.raises(ValueError, match="laufende"):
            service.restore_previous()
        assert registry.get_registry_state(models_dir)["active_version"] == "initial"
        assert artifact_bytes(models_dir) == before
    finally:
        release.set()
    service.wait(30)
    ready = service.get_status()
    assert ready["phase"] == "ready", ready["message"]
    assert ready["can_adopt"]
    assert ready["active_version"] == "initial"
    assert ready["comparison"]["data_report"]["confirmed_count"] == 1
    assert artifact_bytes(models_dir) == before
    assert not (models_dir / "active_model.json").exists()


def test_candidate_survives_restart_and_explicit_adoption_and_rollback_preserve_initial(
    training_workspace
):
    models_dir, _, observations, create = training_workspace
    before = artifact_bytes(models_dir)
    add_observation(observations)
    first_service = create()
    ready = train_ready(first_service)
    candidate_id = ready["candidate_id"]
    paths = registry.version_paths(candidate_id, models_dir)
    assert all(path.is_file() for path in paths.values())
    assert json.loads(first_service.pending_path.read_text(encoding="utf-8"))["candidate_id"] == candidate_id
    first_service.close()
    restarted = create()
    restored_candidate = restarted.get_status()
    assert restored_candidate["phase"] == "ready"
    assert restored_candidate["candidate_id"] == candidate_id
    assert restored_candidate["can_adopt"]
    assert restored_candidate["active_version"] == "initial"
    adopted = restarted.adopt_candidate()
    assert adopted == {"active_version": candidate_id, "previous_version": "initial"}
    status = restarted.get_status()
    assert status["active_version"] == candidate_id
    assert status["new_confirmed_count"] == 0
    assert not status["can_train"]
    assert not status["can_adopt"]
    assert status["can_rollback"]
    assert not restarted.pending_path.exists()
    assert artifact_bytes(models_dir) == before
    for key, contents in before.items():
        assert registry.version_paths("initial", models_dir)[key].read_bytes() == contents
    rollback = restarted.restore_previous()
    assert rollback == {"active_version": "initial", "previous_version": candidate_id}
    assert restarted.get_status()["new_confirmed_count"] == 1
    assert artifact_bytes(models_dir) == before
    assert all(path.is_file() for path in paths.values())


@pytest.mark.parametrize("correction", ["", "Gentoo", "remove_row"])
def test_removed_or_corrected_confirmations_invalidate_pending_candidate(
    training_workspace, correction
):
    models_dir, _, observations, create = training_workspace
    add_observation(observations)
    service = create()
    train_ready(service)
    data = pd.read_csv(observations, keep_default_na=False, float_precision="round_trip")
    if correction == "remove_row":
        data = data.iloc[:0]
    else:
        data.loc[0, "validated_species"] = correction
    data.to_csv(observations, index=False)
    status = service.get_status()
    assert not status["can_adopt"]
    with pytest.raises(ValueError, match="keine geprüfte"):
        service.adopt_candidate()
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"


def test_reference_changes_after_training_invalidate_candidate_even_before_first_adoption(
    training_workspace
):
    models_dir, reference, observations, create = training_workspace
    add_observation(observations)
    service = create()
    train_ready(service)
    data = pd.read_csv(reference, float_precision="round_trip")
    data.loc[0, "bill_length_mm"] += 0.1
    data.to_csv(reference, index=False)
    assert not service.get_status()["can_adopt"]
    with pytest.raises(ValueError):
        service.adopt_candidate()
    assert registry.get_registry_state(models_dir)["active_version"] == "initial"


def test_failed_training_keeps_original_model_and_never_offers_adoption(
    training_workspace, monkeypatch
):
    models_dir, _, observations, create = training_workspace
    before = artifact_bytes(models_dir)
    add_observation(observations)
    service = create()

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("Erwarteter Trainingsfehler")

    monkeypatch.setattr(retraining, "train_and_evaluate", fail_evaluation)
    service.start_training()
    service.wait(30)
    status = service.get_status()
    assert status["phase"] == "error"
    assert "Erwarteter Trainingsfehler" in status["message"]
    assert not status["can_adopt"]
    assert status["can_train"]
    assert status["active_version"] == "initial"
    assert artifact_bytes(models_dir) == before
    assert not service.pending_path.exists()
    assert not (models_dir / "active_model.json").exists()


def test_predictions_without_confirmation_never_trigger_evaluation(training_workspace, monkeypatch):
    models_dir, _, observations, create = training_workspace
    add_observation(observations, confirmed=None)
    service = create()
    calls = []

    def unexpected_evaluation(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Unbestätigte Daten dürfen kein Training auslösen")

    monkeypatch.setattr(retraining, "train_and_evaluate", unexpected_evaluation)
    status = service.get_status()
    assert status["data_report"]["ignored_unconfirmed"] == 1
    assert status["new_confirmed_count"] == 0
    assert not status["can_train"]
    with pytest.raises(ValueError, match="Keine neuen"):
        service.start_training()
    assert not calls
    assert not (models_dir / "versions").exists()


def test_next_training_uses_active_snapshot_with_exact_decimal_roundtrip(training_workspace):
    models_dir, _, observations, create = training_workspace
    add_observation(observations)
    service = create()
    first = train_ready(service)
    service.adopt_candidate()
    add_observation(observations, confirmed="Gentoo", length=51.12345678901235)
    assert service.get_status()["new_confirmed_count"] == 1
    second = train_ready(service)
    assert second["active_version"] == first["candidate_id"]
    assert second["comparison"]["data_report"]["confirmed_count"] == 2
    assert second["comparison"]["baseline_training_size"] == 257
    assert second["comparison"]["candidate_training_size"] == 258
    service.adopt_candidate()
    state = registry.get_registry_state(models_dir)
    assert state["active_version"] == second["candidate_id"]
    assert state["previous_version"] == first["candidate_id"]
    assert not service.get_status()["can_train"]
