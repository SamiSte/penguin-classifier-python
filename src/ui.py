"""Oberflächenelemente und Ergebnisdarstellung der Dash-Anwendung."""

from decimal import Decimal, ROUND_HALF_UP

from dash import dcc, html

from src.figures import FEATURE_LABELS, PLOT_FEATURES, create_penguin_scatter
from src.formatting import format_number
from src.model_service import NUMERIC_LABELS
from src.storage import ALLOWED_SPECIES


NUMERIC_UNITS = {
    "bill_length_mm": "mm", "bill_depth_mm": "mm",
    "flipper_length_mm": "mm", "body_mass_g": "g",
}
NUMERIC_STEPS = {
    "bill_length_mm": 0.1, "bill_depth_mm": 0.1,
    "flipper_length_mm": 1, "body_mass_g": 50,
}
FORM_FIELDS = [*NUMERIC_LABELS, "sex", "island"]
DEFAULT_CHART_HELP = "Messwerte anzeigen: mit der Maus über einen Punkt fahren."


def numeric_default(feature, feature_ranges):
    """Runde den Median auf die für das Merkmal vorgesehene Genauigkeit."""
    precision = Decimal("0.1") if NUMERIC_STEPS[feature] < 1 else Decimal("1")
    return float(Decimal(str(feature_ranges[feature]["median"])).quantize(
        precision, rounding=ROUND_HALF_UP,
    ))


def create_numeric_input(feature, feature_ranges):
    """Explizite, tastaturbedienbare Schrittknöpfe neben freier Zahleneingabe."""
    limits = feature_ranges[feature]
    label, unit = NUMERIC_LABELS[feature], NUMERIC_UNITS[feature]
    step_label = format_number(NUMERIC_STEPS[feature])
    return html.Div([
        html.Label(f"{label} ({unit})", htmlFor=feature),
        html.Div([
            html.Button(
                "−", id=f"{feature}-minus", n_clicks=0,
                className="step-button", type="button",
                title=f"Um {step_label} {unit} verringern",
                **{"aria-label": f"{label} um {step_label} {unit} verringern"},
            ),
            dcc.Input(
                id=feature, type="number", value=numeric_default(feature, feature_ranges),
                min=0, step="any", debounce=False, className="numeric-input",
            ),
            html.Button(
                "+", id=f"{feature}-plus", n_clicks=0,
                className="step-button", type="button",
                title=f"Um {step_label} {unit} erhöhen",
                **{"aria-label": f"{label} um {step_label} {unit} erhöhen"},
            ),
        ], className="number-control"),
        html.Small(
            f"Trainingsbereich: {format_number(limits['minimum'])}–{format_number(limits['maximum'])} {unit}",
            id=f"{feature}-range", className="field-help",
        ),
    ], className="field")


def step_heading(number, text):
    return html.H2([html.Span(f"{number}.", className="step-number"), text])


def empty_prediction(changed=False):
    return html.Div([
        html.Strong("Eingaben geändert" if changed else "Noch keine Vorhersage"),
        html.P(
            "Bitte erneut die Pinguinart bestimmen."
            if changed else "Messwerte prüfen und auf „Pinguinart bestimmen“ klicken."
        ),
    ], className="prediction-placeholder")


def prediction_view(result, island):
    """Modellwahrscheinlichkeiten sind keine garantierte Ergebnissicherheit."""
    probabilities = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    return html.Div([
        html.Div([
            html.Span("Vorhergesagte Art", className="result-label"),
            html.Strong(result["predicted_species"], className="species-name"),
            html.Span(f"Fundort: {island}", className="result-island"),
        ], className="result-summary"),
        html.Div([
            html.Div([html.Span(species), html.Strong(f"{format_number(probability * 100, 1)} %")],
                     className="probability-cell") for species, probability in probabilities
        ], className="probability-grid"),
        html.Small("Modellschätzung; die tatsächliche Art muss fachlich geprüft werden.", className="model-note"),
    ], className="prediction-result")


def comparison_view(metrics):
    if not metrics:
        return ""
    names = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Cohen’s κ", "cohen_kappa")]
    worse = any(metrics["candidate"][key] < metrics["baseline"][key] - 1e-9 for _, key in names)
    return html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th("Kennzahl"), html.Th("Bisherige Variante"), html.Th("Neue Variante")])),
            html.Tbody([html.Tr([
                html.Td(label), html.Td(format_number(metrics['baseline'][key], 3)),
                html.Td(format_number(metrics['candidate'][key], 3)),
            ]) for label, key in names]),
        ], className="metrics-table"),
        html.Small(f"Beide Varianten: gleicher Prüfanteil mit {metrics['validation_size']} Beobachtungen",
                   className="comparison-note"),
        html.P("Mindestens eine Kennzahl ist schlechter. Die bisherige Version kann beibehalten werden.",
               className="comparison-warning") if worse else None,
    ])


def training_details_view(status):
    report = status.get("data_report")
    content = [html.P(f"Aktive Version: {status.get('active_version', 'nicht lesbar')}")]
    if report:
        content.extend([
            html.P("Bestätigte Zusatzdaten: " + ", ".join(
                f"{species}: {report['confirmed_class_counts'][species]}" for species in ALLOWED_SPECIES)),
            html.P("Gesamte Klassenverteilung: " + ", ".join(
                f"{species}: {report['class_counts'][species]}" for species in ALLOWED_SPECIES)),
            html.P(f"Unbestätigt ausgeschlossen: {report['ignored_unconfirmed']} · "
                   f"Dubletten entfernt: {report['duplicates_removed']}"),
            *[html.P(warning) for warning in report["warnings"]],
        ])
    metrics = status.get("comparison")
    if metrics:
        for key, label in (("baseline_parameters", "Bisherige Variante"),
                           ("candidate_parameters", "Neue Variante")):
            parameters = metrics.get(key)
            if parameters:
                depth = parameters["max_depth"] or "unbegrenzt"
                content.append(html.P(
                    f"{label}: {parameters['n_estimators']} Bäume, maximale Tiefe {depth}, "
                    f"mindestens {parameters['min_samples_leaf']} Beobachtung je Blatt."
                ))
        content.extend([
            html.P("Beide Vergleichsmodelle wurden separat trainiert. Die Prüfdaten waren von beiden "
                   "Trainingsläufen ausgeschlossen. Die angezeigten Kennzahlen beziehen sich auf diese "
                   "Vergleichsmodelle; das Auslieferungsmodell nutzt danach alle geprüften Daten."),
            html.P("Fünffache Kreuzvalidierung der neuen Variante (Mittelwert ± Standardabweichung):"),
            html.Ul([html.Li(f"{label}: {format_number(metrics['cross_validation'][key]['mean'], 3)} ± "
                            f"{format_number(metrics['cross_validation'][key]['standard_deviation'], 3)}")
                     for label, key in [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"),
                                       ("Cohen’s κ", "cohen_kappa")]]),
            html.P("Konfusionsmatrizen: Zeilen = tatsächliche Art, Spalten = vorhergesagte Art."),
        ])
        for key, title in [("baseline", "Bisherige Variante"), ("candidate", "Neue Variante")]:
            content.append(html.Table([
                html.Caption(title),
                html.Thead(html.Tr([html.Th("Art"), *[html.Th(s) for s in metrics["classes"]]])),
                html.Tbody([html.Tr([html.Th(species), *[html.Td(n) for n in row]])
                            for species, row in zip(metrics["classes"], metrics[key]["confusion_matrix"])]),
            ], className="metrics-table"))
        content.append(html.P("Wiederholte Vergleiche auf diesem kleinen Prüfanteil ersetzen keine "
                              "unabhängige Bewertung mit neuen Erhebungen."))
    return content


def build_layout(metadata, reference_data, active_version):
    """Erzeuge die Oberfläche; Fachlogik und Callbacks liegen getrennt."""
    return html.Div([
        dcc.Store(id="prediction-store"),
        dcc.Store(id="saved-prediction-id"),
        dcc.Store(id="save-busy", data=False),
        dcc.Store(id="active-model-version", data=active_version),
        dcc.Store(id="training-action"),
        dcc.Interval(id="training-poll", interval=2000, n_intervals=0),
        html.Header([
            html.Div([
                html.H1("Pinguin-Klassifikator"),
            html.P("Messwerte erfassen, Art bestimmen, Beobachtung speichern.", className="header-description"),
            ]),
        html.P(f"Palmer Penguins · {len(reference_data)} Referenzbeobachtungen", className="header-reference"),
        ], className="app-header"),
        html.Main([
            html.Div([
                html.Section([
                    step_heading(1, "Messwerte eingeben"),
                html.P("Die Beispielwerte durch Ihre Messung ersetzen.",
                           className="section-help"),
                    html.Div([create_numeric_input(feature, metadata["feature_ranges"]) for feature in NUMERIC_LABELS],
                             className="measurement-grid"),
                    html.Div([
                        html.Div([
                            html.Label("Geschlecht", htmlFor="sex"),
                            dcc.Dropdown(id="sex", options=[
                                {"label": "Weiblich", "value": "female"},
                                {"label": "Männlich", "value": "male"},
                                {"label": "Unbekannt", "value": "unknown"},
                            ], value="unknown", clearable=False, searchable=False),
                        ], className="field"),
                        html.Div([
                            html.Label("Fundort / Insel", htmlFor="island"),
                            dcc.Dropdown(id="island", options=[
                                {"label": island, "value": island} for island in metadata["allowed_values"]["island"]
                            ], value=metadata["allowed_values"]["island"][0], clearable=False, searchable=False),
                        ], className="field"),
                    ], className="context-grid"),
                html.Small("Der Fundort wird gespeichert, aber nicht für die Vorhersage verwendet.",
                               className="context-help"),
                    html.Button("Pinguinart bestimmen", id="classify-button", n_clicks=0,
                                className="primary-button", type="button"),
                ], className="panel input-panel"),
                html.Section([
                    step_heading(2, "Ergebnis prüfen und speichern"),
                    dcc.Loading(html.Div(id="prediction-output", children=empty_prediction(),
                                        **{"aria-live": "polite"}), type="dot", delay_show=250),
                    html.Div(id="warning-output", **{"aria-live": "polite"}),
                    html.Div([
                        html.Div([
                            html.Label("Fachlich bestätigte Art (optional)",
                                       htmlFor="validated-species"),
                            dcc.Dropdown(
                                id="validated-species",
                                options=[{"label": "Nicht bestätigt", "value": ""}] + [
                                    {"label": species, "value": species}
                                    for species in ALLOWED_SPECIES
                                ],
                                value="", disabled=True, clearable=False, searchable=False,
                            ),
                        ], className="field"),
                        html.Small(
                        "Nur angeben, wenn die Art unabhängig von der Vorhersage bekannt ist.",
                            className="confirmation-help",
                        ),
                    ], className="confirmation-row"),
                    html.Div([
                        html.Button("Beobachtung speichern", id="save-button", n_clicks=0,
                                    disabled=True, className="save-button", type="button"),
                    html.Small("Messwerte und Ergebnis lokal speichern",
                                   className="save-help"),
                    ], className="save-row"),
                    html.Div(id="save-output", **{"aria-live": "polite"}),
                ], className="panel result-panel"),
            ], className="form-column"),
            html.Div([html.Section([
                step_heading(3, "Referenzdaten vergleichen"),
            html.P("Punkte: Referenzdaten. Stern: Ihre aktuelle Beobachtung.",
                       className="section-help"),
                html.Div([
                    html.Div([
                        html.Label(label, htmlFor=axis_id),
                        dcc.Dropdown(id=axis_id, options=[
                            {"label": FEATURE_LABELS[feature], "value": feature}
                            for feature in PLOT_FEATURES
                        ], value=default, clearable=False, searchable=False),
                    ], className="field")
                    for label, axis_id, default in [
                        ("X-Achse", "x-axis-feature", "bill_length_mm"),
                        ("Y-Achse", "y-axis-feature", "bill_depth_mm"),
                    ]
                ], className="axes-grid"),
                dcc.Graph(id="penguin-scatter", figure=create_penguin_scatter(
                    reference_data, "bill_length_mm", "bill_depth_mm",
                ), config={"displaylogo": False, "responsive": True, "displayModeBar": False},
                    responsive=True, className="comparison-graph"),
                html.P(DEFAULT_CHART_HELP, id="chart-help", className="chart-help",
                       **{"aria-live": "polite"}),
            ], className="panel chart-panel"),
            html.Section([
                step_heading(4, "Modell aktualisieren"),
                html.Div(id="training-overview", className="training-overview"),
                html.Div(id="training-status", role="status", **{"aria-live": "polite"}),
                html.Div(id="training-comparison"),
                html.Div([
                    html.Button("Re-Training starten", id="train-button", n_clicks=0,
                                disabled=True, className="save-button", type="button"),
                    html.Button("Neue Version übernehmen", id="adopt-button", n_clicks=0,
                                disabled=True, className="save-button", type="button"),
                ], className="training-actions"),
                html.Div(id="training-action-error", **{"aria-live": "polite"}),
                html.Details([
                    html.Summary("Prüfdetails und vorherige Version"),
                    html.Div(id="training-details"),
                    html.Button("Vorherige Version wiederherstellen", id="rollback-button",
                                n_clicks=0, disabled=True, className="save-button", type="button"),
                ], className="training-details"),
            ], className="panel training-panel")], className="analysis-column"),
        ], className="workspace"),
    ], className="app-shell")
