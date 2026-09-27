"""Regressionen für die Benutzeraktionen ohne Schreibzugriff auf Nutzerdaten."""

from pathlib import Path
from functools import partial
from graphlib import TopologicalSorter
from types import SimpleNamespace

import pandas as pd
import pytest

import app as ui
from src.storage import ALLOWED_SPECIES


@pytest.fixture
def values():
    return [47.5, 14.5, 218.0, 5100.0, "unknown", "Biscoe"]


@pytest.fixture
def prediction(monkeypatch, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="classify-button"))
    return ui.classify_penguin(1, *values)[2]


def test_default_bill_length_is_rounded_to_one_decimal():
    assert ui.numeric_default("bill_length_mm", ui.FEATURE_RANGES) == 44.5


def test_successful_prediction_enables_save_and_adds_star(prediction, values):
    assert ui.toggle_save_button(prediction, None, False, *values) == (False, False)
    figure = ui.update_penguin_scatter("bill_length_mm", "bill_depth_mm", prediction, *values)
    assert figure.data[-1].name == "Neue Beobachtung"
    assert figure.data[-1].x[0] == values[0]


@pytest.mark.parametrize("index,replacement", [
    (0, 48.1), (1, 15), (2, 220), (3, 5200), (4, "female"), (5, "Dream"),
])
def test_any_changed_input_invalidates_result_and_disables_save(
    monkeypatch, prediction, values, index, replacement,
):
    values[index] = replacement
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id=ui.FORM_FIELDS[index]))
    content, warning, stored, confirmed = ui.classify_penguin(1, *values)
    assert stored is None
    assert confirmed == ""
    assert warning == ""
    assert content.children[0].children == "Eingaben geändert"
    assert ui.toggle_save_button(prediction, None, False, *values) == (True, True)
    figure = ui.update_penguin_scatter("bill_length_mm", "bill_depth_mm", prediction, *values)
    assert all(trace.name != "Neue Beobachtung" for trace in figure.data)


def test_save_rejects_changed_input_even_before_store_is_cleared(monkeypatch, prediction, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))
    writes = []
    monkeypatch.setattr(ui, "save_observation", lambda **kwargs: writes.append(kwargs))
    values[0] = 52
    content, saved_id = ui.save_classified_observation(1, prediction, None, "Adelie", *values)
    assert not writes
    assert saved_id is None
    assert "zuerst klassifizieren" in content.children


def test_successful_save_is_disabled_and_not_written_twice(monkeypatch, prediction, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))
    writes = []

    def save(**kwargs):
        writes.append(kwargs)
        return Path("new_observations.csv")

    monkeypatch.setattr(ui, "save_observation", save)
    _, saved_id = ui.save_classified_observation(1, prediction, None, "", *values)
    assert saved_id == prediction["prediction_id"]
    assert ui.toggle_save_button(prediction, saved_id, False, *values) == (True, True)
    ui.save_classified_observation(2, prediction, saved_id, "Adelie", *values)
    assert len(writes) == 1


def test_save_error_keeps_prediction_available_for_retry(monkeypatch, prediction, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))

    def fail(**kwargs):
        raise OSError("Datei gesperrt")

    monkeypatch.setattr(ui, "save_observation", fail)
    _, saved_id = ui.save_classified_observation(1, prediction, None, "Adelie", *values)
    assert saved_id is None
    assert ui.toggle_save_button(prediction, saved_id, False, *values) == (False, False)
    assert ui.toggle_save_button(prediction, saved_id, True, *values) == (True, True)


@pytest.mark.parametrize("invalid_value", [None, 0, -1])
def test_invalid_direct_input_shows_error_and_drops_prediction(monkeypatch, values, invalid_value):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="classify-button"))
    values[0] = invalid_value
    content, _, stored, confirmed = ui.classify_penguin(1, *values)
    assert stored is None
    assert confirmed == ""
    assert content.className == "message error-message"


def test_out_of_range_input_preserves_warning_and_prediction(monkeypatch, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="classify-button"))
    values[0] = 70
    _, warning, stored, _ = ui.classify_penguin(1, *values)
    assert stored is not None
    assert warning.className == "message warning-message"


def test_dash_layout_and_callback_registration():
    client = ui.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200


def test_callback_dependencies_have_no_cycles():
    """Auch bedingt unveränderte Outputs zählen zum statischen Dash-Graphen."""
    graph = TopologicalSorter()
    for callback in ui.app.callback_map.values():
        outputs = callback["output"]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        # State liest nur mit und löst den Callback selbst nicht aus.
        inputs = [f"{item['id']}.{item['property']}" for item in callback["inputs"]]
        for output in outputs:
            graph.add(f"{output.component_id}.{output.component_property}", *inputs)
    tuple(graph.static_order())


def test_classification_never_preselects_its_prediction_as_confirmed_species(monkeypatch, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="classify-button"))
    _, _, stored, confirmed = ui.classify_penguin(1, *values)
    assert stored["predicted_species"] in ALLOWED_SPECIES
    assert confirmed == ""


@pytest.mark.parametrize("confirmed", ["", "Adelie"])
def test_ui_saves_independent_confirmation_in_csv(
    monkeypatch, prediction, values, tmp_path, confirmed,
):
    """Die tatsächliche Speicherkette erhält Modellschätzung und Fachlabel getrennt."""
    from src.storage import save_observation

    path = tmp_path / "observations.csv"
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))
    monkeypatch.setattr(ui, "save_observation", partial(save_observation, output_path=path))
    content, saved_id = ui.save_classified_observation(1, prediction, None, confirmed, *values)

    data = pd.read_csv(path, keep_default_na=False)
    assert saved_id == prediction["prediction_id"]
    assert data.loc[0, "predicted_species"] == prediction["predicted_species"]
    assert data.loc[0, "validated_species"] == confirmed
    assert data.loc[0, "probability_gentoo"] == prediction["probabilities"]["Gentoo"]
    if confirmed:
        assert confirmed != prediction["predicted_species"]
        assert f"Fachlich bestätigte Art: {confirmed}" in content.children
    else:
        assert "Art nicht fachlich bestätigt" in content.children


def test_invalid_confirmation_shows_error_without_saving(monkeypatch, prediction, values, tmp_path):
    from src.storage import save_observation

    path = tmp_path / "observations.csv"
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))
    monkeypatch.setattr(ui, "save_observation", partial(save_observation, output_path=path))
    content, saved_id = ui.save_classified_observation(1, prediction, None, "Emperor", *values)
    assert content.className == "message error-message"
    assert saved_id is None
    assert not path.exists()
    assert ui.toggle_save_button(prediction, saved_id, False, *values) == (False, False)


def test_model_change_clears_prediction_and_confirmation(monkeypatch, values):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="active-model-version"))
    content, _, stored, confirmed = ui.classify_penguin(1, *values, "new-version")
    assert content.children[0].children == "Modell aktualisiert"
    assert stored is None and confirmed == ""


def test_stale_model_prediction_cannot_be_saved_or_plotted(monkeypatch, prediction, values):
    monkeypatch.setattr(ui, "get_registry_state", lambda: {"active_version": "new-version"})
    assert ui.toggle_save_button(prediction, None, False, *values) == (True, True)
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="save-button"))
    writes = []
    monkeypatch.setattr(ui, "save_observation", lambda **kwargs: writes.append(kwargs))
    _, saved_id = ui.save_classified_observation(1, prediction, None, "Gentoo", *values)
    assert saved_id is None and not writes
    figure = ui.update_penguin_scatter("bill_length_mm", "bill_depth_mm", prediction, *values)
    assert all(trace.name != "Neue Beobachtung" for trace in figure.data)


@pytest.mark.parametrize("button,action", [
    ("train-button", "start_training"), ("adopt-button", "adopt_candidate"),
    ("rollback-button", "restore_previous"),
])
def test_training_buttons_invoke_only_the_requested_action(monkeypatch, button, action):
    calls = []
    fake = SimpleNamespace(**{name: (lambda name=name: calls.append(name))
                             for name in ("start_training", "adopt_candidate", "restore_previous")})
    monkeypatch.setattr(ui, "RETRAINING", fake)
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id=button))
    token, error = ui.handle_training_action(1, 1, 1)
    assert token and error == ""
    assert calls == [action]


def test_training_error_is_visible(monkeypatch):
    def fail():
        raise ValueError("Keine bestätigten Daten vorhanden.")
    fake = SimpleNamespace(start_training=fail, adopt_candidate=fail, restore_previous=fail)
    monkeypatch.setattr(ui, "RETRAINING", fake)
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id="train-button"))
    _, error = ui.handle_training_action(1, 0, 0)
    assert "Keine bestätigten Daten" in error.children
    assert error.className == "message error-message"


def test_polling_does_not_reset_prediction_if_version_is_unchanged(monkeypatch):
    status = {"new_confirmed_count": 0, "data_report": None, "phase": "idle", "can_train": False,
              "can_adopt": False, "can_rollback": False, "active_version": "initial",
              "message": "Bereit", "data_error": None}
    monkeypatch.setattr(ui, "RETRAINING", SimpleNamespace(get_status=lambda: status))
    result = ui.refresh_training(1, None, "initial")
    assert result[-1] is ui.no_update
    assert result[4:7] == (True, True, True)
    status["active_version"] = "new-version"
    assert ui.refresh_training(2, None, "initial")[-1] == "new-version"
