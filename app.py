"""Dash-Webanwendung zur Klassifikation von Pinguinarten."""

from uuid import uuid4

from dash import Dash, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from src.data_processing import load_clean_data
from src.figures import create_penguin_scatter
from src.formatting import format_number
from src.model_service import load_metadata, predict_species
from src.model_registry import get_registry_state
from src.retraining import RetrainingService
from src.storage import save_observation
from src.ui import (
    FORM_FIELDS, NUMERIC_LABELS, NUMERIC_STEPS, NUMERIC_UNITS, build_layout,
    comparison_view, empty_prediction, numeric_default, prediction_view, training_details_view,
)


METADATA = load_metadata()
FEATURE_RANGES = METADATA["feature_ranges"]
REFERENCE_DATA = load_clean_data()
RETRAINING = RetrainingService()


app = Dash(__name__, title="Pinguin-Klassifikator", update_title=None)
server = app.server

app.layout = build_layout(METADATA, REFERENCE_DATA, get_registry_state()["active_version"])


# Im Browser synchron: Auch schnelle Mehrfachklicks verwenden den letzten Wert.
# step='any' lässt manuelle Messwerte zu, ohne versteckte HTML-Step-Validierung.
for feature, step in NUMERIC_STEPS.items():
    app.clientside_callback(
        f"""function(minus, plus, value) {{
            const trigger = dash_clientside.callback_context.triggered_id;
            if (!trigger) return dash_clientside.no_update;
            return window.penguinUi.adjustValue(value, {step}, {numeric_default(feature, FEATURE_RANGES)},
                trigger.endsWith('-plus') ? 1 : -1);
        }}""",
        Output(feature, "value"), Input(f"{feature}-minus", "n_clicks"),
        Input(f"{feature}-plus", "n_clicks"), State(feature, "value"),
        prevent_initial_call=True,
    )


def prediction_matches_form(stored_prediction, values):
    """Veraltete Vorhersagen auch direkt vor Speichern/Zeichnen abweisen."""
    return (bool(stored_prediction)
            and stored_prediction.get("form_values") == list(values[:len(FORM_FIELDS)])
            and stored_prediction.get("model_version") == get_registry_state()["active_version"])


@app.callback(
    Output("prediction-output", "children"), Output("warning-output", "children"),
    Output("prediction-store", "data"), Output("validated-species", "value"),
    Input("classify-button", "n_clicks"),
    *[Input(field, "value") for field in FORM_FIELDS],
    Input("active-model-version", "data"), prevent_initial_call=True,
)
def classify_penguin(n_clicks, *values):
    """Neue Eingaben/Vorhersagen verwerfen auch die bisherige Artbestätigung."""
    if ctx.triggered_id == "active-model-version":
        return html.Div([
            html.Strong("Modell aktualisiert"), html.P("Bitte erneut die Pinguinart bestimmen."),
        ], className="prediction-placeholder"), "", None, ""
    if ctx.triggered_id != "classify-button":
        return empty_prediction(changed=bool(n_clicks)), "", None, ""
    if not n_clicks:
        raise PreventUpdate
    values = values[:len(FORM_FIELDS)]
    form = dict(zip(FORM_FIELDS, values))
    observation = {feature: form[feature] for feature in FORM_FIELDS if feature != "island"}
    try:
        result = predict_species(observation)
    except (ValueError, FileNotFoundError, OSError) as error:
        return html.Div([
            html.Strong("Bitte Eingaben prüfen" if isinstance(error, ValueError) else "Technischer Fehler"),
            html.P(str(error)),
        ], className="message error-message"), "", None, ""
    warning = html.Div([
        html.Strong("Außerhalb des Trainingsbereichs"),
        html.Ul([html.Li(text) for text in result["warnings"]]),
    ], className="message warning-message") if result["warnings"] else ""
    stored = {
        "prediction_id": uuid4().hex,
        "observation": result["validated_observation"],
        "predicted_species": result["predicted_species"],
        "probabilities": result["probabilities"],
        "model_version": result["model_version"],
        "island": form["island"], "form_values": list(values),
    }
    return prediction_view(result, form["island"]), warning, stored, ""


@app.callback(
    Output("save-button", "disabled"), Output("validated-species", "disabled"),
    Input("prediction-store", "data"),
    Input("saved-prediction-id", "data"), Input("save-busy", "data"),
    *[Input(field, "value") for field in FORM_FIELDS],
    Input("active-model-version", "data"),
)
def toggle_save_button(stored_prediction, saved_prediction_id, save_busy, *values):
    """Bestätigung und Speichern nur für eine aktuelle, ungespeicherte Vorhersage."""
    disabled = (bool(save_busy) or not prediction_matches_form(stored_prediction, values)
                or stored_prediction["prediction_id"] == saved_prediction_id)
    return disabled, disabled


@app.callback(
    Output("save-output", "children"), Output("saved-prediction-id", "data"),
    Input("save-button", "n_clicks"), Input("prediction-store", "data"),
    State("saved-prediction-id", "data"), State("validated-species", "value"),
    *[State(field, "value") for field in FORM_FIELDS],
    prevent_initial_call=True,
    running=[(Output("save-busy", "data"), True, False)],
)
def save_classified_observation(
    n_clicks, stored_prediction, saved_prediction_id, validated_species, *values,
):
    """Nur die aktuelle Vorhersage speichern und doppelte Speicherung abweisen."""
    if ctx.triggered_id != "save-button":
        return "", no_update
    if not n_clicks:
        raise PreventUpdate
    if not prediction_matches_form(stored_prediction, values):
        return html.Div("Bitte die aktuellen Eingaben zuerst klassifizieren.",
                        className="message error-message"), saved_prediction_id
    prediction_id = stored_prediction["prediction_id"]
    if prediction_id == saved_prediction_id:
        return html.Div("Diese Beobachtung wurde bereits gespeichert.",
                        className="save-status"), saved_prediction_id
    try:
        save_observation(
            observation=stored_prediction["observation"], island=stored_prediction["island"],
            predicted_species=stored_prediction["predicted_species"],
            probabilities=stored_prediction["probabilities"],
            validated_species=validated_species,
        )
    except (KeyError, OSError, ValueError) as error:
        return html.Div([html.Strong("Speichern nicht möglich. "), str(error)],
                        className="message error-message"), saved_prediction_id
    confirmation = (f"Fachlich bestätigte Art: {validated_species}."
                    if validated_species else "Art nicht fachlich bestätigt.")
    return html.Div(f"Gespeichert. {confirmation}", className="save-status"), prediction_id


@app.callback(
    Output("penguin-scatter", "figure"), Input("x-axis-feature", "value"),
    Input("y-axis-feature", "value"), Input("prediction-store", "data"),
    *[Input(field, "value") for field in FORM_FIELDS],
    Input("active-model-version", "data"),
)
def update_penguin_scatter(x_feature, y_feature, stored_prediction, *values):
    current = stored_prediction if prediction_matches_form(stored_prediction, values) else None
    return create_penguin_scatter(
        reference_data=REFERENCE_DATA, x_feature=x_feature, y_feature=y_feature,
        new_observation=current["observation"] if current else None,
        predicted_species=current["predicted_species"] if current else None,
    )


@app.callback(
    Output("training-overview", "children"), Output("training-status", "children"),
    Output("training-comparison", "children"), Output("training-details", "children"),
    Output("train-button", "disabled"), Output("adopt-button", "disabled"),
    Output("rollback-button", "disabled"), Output("active-model-version", "data"),
    Input("training-poll", "n_intervals"),
    Input("training-action", "data"), State("active-model-version", "data"),
)
def refresh_training(_poll, _action, known_version):
    # Neue CSV-Einträge werden per Intervall eingelesen. Ein Speicher-Input hier
    # würde über die Modellversion und Vorhersage einen Callback-Kreis erzeugen.
    status = RETRAINING.get_status()
    count = status["new_confirmed_count"]
    noun = "Beobachtung" if count == 1 else "Beobachtungen"
    overview = f"{count} neue bestätigte {noun} seit der letzten Übernahme"
    message = status.get("data_error") or status["message"]
    if (not status.get("data_error") and status["phase"] == "idle" and not status["can_train"]):
        message = "Zum Re-Training zunächst eine neue Beobachtung mit fachlich bestätigter Art speichern."
    class_name = "training-message"
    if status["phase"] == "training":
        class_name += " is-training"
    if status.get("data_error") or status["phase"] == "error":
        class_name += " training-error"
    version = status.get("active_version", known_version)
    return (overview, html.Div(message, className=class_name), comparison_view(status.get("comparison")),
            training_details_view(status), not status["can_train"], not status["can_adopt"],
            not status["can_rollback"], version if version != known_version else no_update)


@app.callback(
    Output("training-action", "data"), Output("training-action-error", "children"),
    Input("train-button", "n_clicks"), Input("adopt-button", "n_clicks"),
    Input("rollback-button", "n_clicks"), prevent_initial_call=True,
)
def handle_training_action(_train, _adopt, _rollback):
    actions = {"train-button": RETRAINING.start_training,
               "adopt-button": RETRAINING.adopt_candidate,
               "rollback-button": RETRAINING.restore_previous}
    if ctx.triggered_id not in actions:
        raise PreventUpdate
    try:
        actions[ctx.triggered_id]()
        return uuid4().hex, ""
    except (ValueError, OSError, KeyError) as error:
        return uuid4().hex, html.Div(str(error), className="message error-message")


@app.callback(
    [Output(f"{feature}-range", "children") for feature in NUMERIC_LABELS],
    Input("active-model-version", "data"),
)
def refresh_training_ranges(_version):
    ranges = load_metadata()["feature_ranges"]
    return [f"Trainingsbereich: {format_number(ranges[feature]['minimum'])}–{format_number(ranges[feature]['maximum'])} "
            f"{NUMERIC_UNITS[feature]}" for feature in NUMERIC_LABELS]


if __name__ == "__main__":
    app.run(debug=True, jupyter_mode="external", use_reloader=False)
