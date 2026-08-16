"""Dash-Webanwendung zur Klassifikation von Pinguinarten."""

from uuid import uuid4

from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from src.data_processing import load_clean_data
from src.figures import FEATURE_LABELS, PLOT_FEATURES, create_penguin_scatter
from src.model_service import load_metadata, predict_species
from src.storage import save_observation


# Modellinformationen und Referenzdaten beim Start laden.
METADATA = load_metadata()
FEATURE_RANGES = METADATA["feature_ranges"]
ISLANDS = METADATA["allowed_values"]["island"]
REFERENCE_DATA = load_clean_data()

NUMERIC_LABELS = {
    "bill_length_mm": "Schnabellänge",
    "bill_depth_mm": "Schnabeltiefe",
    "flipper_length_mm": "Flossenlänge",
    "body_mass_g": "Körpergewicht",
}

NUMERIC_UNITS = {
    "bill_length_mm": "mm",
    "bill_depth_mm": "mm",
    "flipper_length_mm": "mm",
    "body_mass_g": "g",
}

NUMERIC_STEPS = {
    "bill_length_mm": 0.1,
    "bill_depth_mm": 0.1,
    "flipper_length_mm": 1,
    "body_mass_g": 50,
}


def create_numeric_input(feature: str) -> html.Div:
    """Erstelle ein beschriftetes Zahlenfeld für ein Körpermerkmal."""

    feature_range = FEATURE_RANGES[feature]
    label = NUMERIC_LABELS[feature]
    unit = NUMERIC_UNITS[feature]

    return html.Div(
        children=[
            html.Label(
                f"{label} ({unit})",
                htmlFor=feature,
                style={
                    "display": "block",
                    "fontWeight": "600",
                    "marginBottom": "6px",
                },
            ),
            dcc.Input(
                id=feature,
                type="number",
                value=feature_range["median"],
                min=0,
                step=NUMERIC_STEPS[feature],
                debounce=True,
                style={
                    "boxSizing": "border-box",
                    "fontSize": "16px",
                    "padding": "10px",
                    "width": "100%",
                },
            ),
            html.Small(
                (
                    "Trainingsbereich: "
                    f"{feature_range['minimum']:g}–"
                    f"{feature_range['maximum']:g} {unit}"
                ),
                style={
                    "color": "#666666",
                    "display": "block",
                    "marginTop": "5px",
                },
            ),
        ],
        style={
            "flex": "1 1 280px",
            "marginBottom": "18px",
        },
    )


def determine_assignment_strength(
    maximum_probability: float,
) -> tuple[str, str]:
    """Ordne die höchste Modellwahrscheinlichkeit heuristisch ein."""

    if maximum_probability >= 0.80:
        return (
            "hoch",
            "Die Messwertkombination wird vom Modell eindeutig zugeordnet.",
        )

    if maximum_probability >= 0.60:
        return (
            "mittel",
            "Die Zuordnung ist plausibel, aber nicht vollständig eindeutig.",
        )

    return (
        "niedrig",
        (
            "Mehrere Arten weisen ähnliche Wahrscheinlichkeiten auf. "
            "Bitte Messwerte prüfen und das Ergebnis vorsichtig interpretieren."
        ),
    )


app = Dash(__name__, title="Pinguin-Klassifikator")
server = app.server

app.layout = html.Div(
    children=[
        dcc.Store(id="prediction-store"),
        dcc.Store(id="saved-prediction-id"),
        html.Header(
            children=[
                html.H1("Pinguin-Klassifikator", style={"marginBottom": "8px"}),
                html.P(
                    (
                        "Bestimmung der Arten Adelie, Chinstrap und Gentoo "
                        "anhand individueller Körpermerkmale."
                    ),
                    style={"fontSize": "17px", "marginTop": "0"},
                ),
            ],
            style={
                "borderBottom": "1px solid #dddddd",
                "marginBottom": "28px",
                "paddingBottom": "18px",
            },
        ),
        html.Main(
            children=[
                html.H2("Neue Beobachtung erfassen"),
                html.Div(
                    children=[
                        create_numeric_input("bill_length_mm"),
                        create_numeric_input("bill_depth_mm"),
                        create_numeric_input("flipper_length_mm"),
                        create_numeric_input("body_mass_g"),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "20px",
                    },
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "Geschlecht",
                                    htmlFor="sex",
                                    style={
                                        "display": "block",
                                        "fontWeight": "600",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="sex",
                                    options=[
                                        {"label": "Weiblich", "value": "female"},
                                        {"label": "Männlich", "value": "male"},
                                        {"label": "Unbekannt", "value": "unknown"},
                                    ],
                                    value="unknown",
                                    clearable=False,
                                ),
                            ],
                            style={
                                "flex": "1 1 280px",
                                "marginBottom": "18px",
                            },
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Fundort / Insel",
                                    htmlFor="island",
                                    style={
                                        "display": "block",
                                        "fontWeight": "600",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="island",
                                    options=[
                                        {"label": island, "value": island}
                                        for island in ISLANDS
                                    ],
                                    value=ISLANDS[0],
                                    clearable=False,
                                ),
                                html.Small(
                                    (
                                        "Der Fundort wird als Kontext erfasst, "
                                        "aber nicht für die Klassifikation verwendet."
                                    ),
                                    style={
                                        "color": "#666666",
                                        "display": "block",
                                        "marginTop": "5px",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1 1 280px",
                                "marginBottom": "18px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "20px",
                    },
                ),
                html.Button(
                    "Pinguinart bestimmen",
                    id="classify-button",
                    n_clicks=0,
                    style={
                        "cursor": "pointer",
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "padding": "12px 20px",
                    },
                ),
                dcc.Loading(
                    children=html.Div(
                        id="prediction-output",
                        style={"marginTop": "28px"},
                    ),
                    type="default",
                ),
                html.Div(id="warning-output", style={"marginTop": "18px"}),
                html.Div(
                    children=[
                        html.Button(
                            "Beobachtung speichern",
                            id="save-button",
                            n_clicks=0,
                            disabled=True,
                            style={
                                "cursor": "pointer",
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "marginTop": "20px",
                                "padding": "10px 18px",
                            },
                        ),
                        html.Small(
                            (
                                "Gespeichert wird die zuletzt erfolgreich "
                                "klassifizierte Beobachtung."
                            ),
                            style={
                                "color": "#666666",
                                "display": "block",
                                "marginTop": "6px",
                            },
                        ),
                        html.Div(
                            id="save-output",
                            style={"marginTop": "12px"},
                        ),
                    ]
                ),
                html.Hr(
                    style={
                        "border": "none",
                        "borderTop": "1px solid #dddddd",
                        "margin": "36px 0",
                    }
                ),
                html.H2("Vergleich mit den Trainingsdaten"),
                html.P(
                    (
                        "Die Punkte zeigen die bereinigten Trainingsdaten. "
                        "Nach einer Klassifikation wird die neue Beobachtung "
                        "als schwarzer Stern dargestellt."
                    )
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "X-Achse",
                                    htmlFor="x-axis-feature",
                                    style={
                                        "display": "block",
                                        "fontWeight": "600",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="x-axis-feature",
                                    options=[
                                        {
                                            "label": FEATURE_LABELS[feature],
                                            "value": feature,
                                        }
                                        for feature in PLOT_FEATURES
                                    ],
                                    value="bill_length_mm",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1 1 280px"},
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Y-Achse",
                                    htmlFor="y-axis-feature",
                                    style={
                                        "display": "block",
                                        "fontWeight": "600",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="y-axis-feature",
                                    options=[
                                        {
                                            "label": FEATURE_LABELS[feature],
                                            "value": feature,
                                        }
                                        for feature in PLOT_FEATURES
                                    ],
                                    value="bill_depth_mm",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1 1 280px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "20px",
                        "marginBottom": "18px",
                    },
                ),
                dcc.Graph(
                    id="penguin-scatter",
                    figure=create_penguin_scatter(
                        reference_data=REFERENCE_DATA,
                        x_feature="bill_length_mm",
                        y_feature="bill_depth_mm",
                    ),
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                ),
            ]
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "lineHeight": "1.5",
        "margin": "0 auto",
        "maxWidth": "900px",
        "padding": "28px 22px 60px",
    },
)


@callback(
    Output("prediction-output", "children"),
    Output("warning-output", "children"),
    Output("prediction-store", "data"),
    Input("classify-button", "n_clicks"),
    State("bill_length_mm", "value"),
    State("bill_depth_mm", "value"),
    State("flipper_length_mm", "value"),
    State("body_mass_g", "value"),
    State("sex", "value"),
    State("island", "value"),
    prevent_initial_call=True,
)
def classify_penguin(
    n_clicks,
    bill_length_mm,
    bill_depth_mm,
    flipper_length_mm,
    body_mass_g,
    sex,
    island,
):
    """Klassifiziere die über die Oberfläche eingegebene Beobachtung."""

    if not n_clicks:
        raise PreventUpdate

    observation = {
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "sex": sex,
    }

    try:
        result = predict_species(observation)
    except ValueError as error:
        return (
            html.Div(
                children=[
                    html.H3("Eingaben konnten nicht verarbeitet werden"),
                    html.P(str(error)),
                ],
                style={
                    "border": "1px solid #cc0000",
                    "padding": "16px",
                },
            ),
            "",
            None,
        )
    except (FileNotFoundError, OSError) as error:
        return (
            html.Div(
                children=[
                    html.H3("Technischer Fehler"),
                    html.P(str(error)),
                ],
                style={
                    "border": "1px solid #cc0000",
                    "padding": "16px",
                },
            ),
            "",
            None,
        )

    predicted_species = result["predicted_species"]
    sorted_probabilities = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    maximum_probability = sorted_probabilities[0][1]
    assignment_strength, assignment_message = determine_assignment_strength(
        maximum_probability
    )

    probability_rows = [
        html.Li(f"{species}: {probability:.1%}")
        for species, probability in sorted_probabilities
    ]

    prediction_content = html.Div(
        children=[
            html.H2(
                f"Vorhergesagte Art: {predicted_species}",
                style={"marginTop": "0"},
            ),
            html.Div(
                children=[
                    html.Strong(
                        "Sicherheit der Modellzuordnung: "
                        f"{assignment_strength.capitalize()}"
                    ),
                    html.P(assignment_message, style={"marginBottom": "0"}),
                ],
                style={
                    "border": "1px solid #bbbbbb",
                    "marginBottom": "18px",
                    "padding": "12px",
                },
            ),
            html.H3("Klassenwahrscheinlichkeiten"),
            html.Ul(probability_rows),
            html.P(
                f"Erfasster Fundort: {island}",
                style={"marginBottom": "0"},
            ),
        ],
        style={
            "border": "1px solid #888888",
            "padding": "20px",
        },
    )

    if result["warnings"]:
        warning_content = html.Div(
            children=[
                html.H3(
                    "Hinweise zu den Eingabewerten",
                    style={"marginTop": "0"},
                ),
                html.Ul([html.Li(warning) for warning in result["warnings"]]),
            ],
            style={
                "border": "1px solid #aa7700",
                "padding": "16px",
            },
        )
    else:
        warning_content = ""

    stored_prediction = {
        "prediction_id": uuid4().hex,
        "observation": result["validated_observation"],
        "predicted_species": predicted_species,
        "probabilities": result["probabilities"],
        "island": island,
    }

    return prediction_content, warning_content, stored_prediction


@callback(
    Output("save-button", "disabled"),
    Input("prediction-store", "data"),
)
def toggle_save_button(stored_prediction):
    """Aktiviere Speichern erst nach erfolgreicher Klassifikation."""

    return not bool(stored_prediction)


@callback(
    Output("save-output", "children"),
    Output("saved-prediction-id", "data"),
    Input("save-button", "n_clicks"),
    State("prediction-store", "data"),
    State("saved-prediction-id", "data"),
    prevent_initial_call=True,
)
def save_classified_observation(
    n_clicks,
    stored_prediction,
    saved_prediction_id,
):
    """Speichere die zuletzt klassifizierte Beobachtung als CSV-Zeile."""

    if not n_clicks:
        raise PreventUpdate

    if not stored_prediction:
        return (
            html.Div(
                "Bitte zuerst eine Beobachtung klassifizieren.",
                style={"color": "#aa0000"},
            ),
            saved_prediction_id,
        )

    prediction_id = stored_prediction["prediction_id"]

    if prediction_id == saved_prediction_id:
        return (
            html.Div(
                "Diese Beobachtung wurde bereits gespeichert.",
                style={"color": "#666666"},
            ),
            saved_prediction_id,
        )

    try:
        saved_path = save_observation(
            observation=stored_prediction["observation"],
            island=stored_prediction["island"],
            predicted_species=stored_prediction["predicted_species"],
            probabilities=stored_prediction["probabilities"],
        )
    except (KeyError, OSError, ValueError) as error:
        return (
            html.Div(
                children=[
                    html.Strong("Speichern nicht möglich."),
                    html.Div(str(error)),
                ],
                style={
                    "border": "1px solid #cc0000",
                    "padding": "12px",
                },
            ),
            saved_prediction_id,
        )

    return (
        html.Div(
            children=[
                html.Strong("Beobachtung erfolgreich gespeichert."),
                html.Div(f"Datei: {saved_path.name}"),
            ],
            style={
                "border": "1px solid #228833",
                "padding": "12px",
            },
        ),
        prediction_id,
    )


@callback(
    Output("penguin-scatter", "figure"),
    Input("x-axis-feature", "value"),
    Input("y-axis-feature", "value"),
    Input("prediction-store", "data"),
)
def update_penguin_scatter(x_feature, y_feature, stored_prediction):
    """Aktualisiere Achsen und gegebenenfalls den neuen Datenpunkt."""

    new_observation = None
    predicted_species = None

    if stored_prediction:
        new_observation = stored_prediction.get("observation")
        predicted_species = stored_prediction.get("predicted_species")

    return create_penguin_scatter(
        reference_data=REFERENCE_DATA,
        x_feature=x_feature,
        y_feature=y_feature,
        new_observation=new_observation,
        predicted_species=predicted_species,
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        jupyter_mode="external",
        use_reloader=False,
    )
