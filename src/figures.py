"""Visualisierungen für die Pinguin-Klassifikationsanwendung."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


FEATURE_LABELS = {
    "bill_length_mm": "Schnabellänge (mm)",
    "bill_depth_mm": "Schnabeltiefe (mm)",
    "flipper_length_mm": "Flossenlänge (mm)",
    "body_mass_g": "Körpergewicht (g)",
}

PLOT_FEATURES = list(FEATURE_LABELS)


def create_penguin_scatter(
    reference_data: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    new_observation: Mapping[str, Any] | None = None,
    predicted_species: str | None = None,
) -> go.Figure:
    """Erstelle ein Streudiagramm der Trainingsdaten.

    Die Referenzbeobachtungen werden nach ihrer tatsächlichen Art eingefärbt.
    Eine neue Beobachtung kann zusätzlich als schwarzer Stern dargestellt
    werden.
    """

    if x_feature not in PLOT_FEATURES:
        raise ValueError(
            f"Unzulässiges Merkmal für die X-Achse: {x_feature}"
        )

    if y_feature not in PLOT_FEATURES:
        raise ValueError(
            f"Unzulässiges Merkmal für die Y-Achse: {y_feature}"
        )

    plot_data = reference_data.copy()
    plot_data["sex"] = plot_data["sex"].fillna("unknown")

    figure = px.scatter(
        plot_data,
        x=x_feature,
        y=y_feature,
        color="species",
        labels={
            **FEATURE_LABELS,
            "species": "Pinguinart",
            "island": "Insel",
            "sex": "Geschlecht",
        },
        hover_data=["island", "sex"],
        category_orders={"species": ["Adelie", "Chinstrap", "Gentoo"]},
        color_discrete_map={
            "Adelie": "#426c91",
            "Chinstrap": "#b47540",
            "Gentoo": "#63806a",
        },
    )

    figure.update_traces(
        marker={
            "size": 7,
            "opacity": 0.75,
        }
    )

    if new_observation is not None:
        missing_features = [
            feature
            for feature in (x_feature, y_feature)
            if feature not in new_observation
        ]

        if missing_features:
            raise ValueError(
                "Für die Darstellung fehlen folgende Merkmale: "
                + ", ".join(missing_features)
            )

        species_text = predicted_species or "Noch nicht klassifiziert"

        figure.add_trace(
            go.Scatter(
                x=[new_observation[x_feature]],
                y=[new_observation[y_feature]],
                mode="markers",
                name="Neue Beobachtung",
                marker={
                    "symbol": "star",
                    "size": 21,
                    "color": "black",
                    "line": {
                        "color": "white",
                        "width": 1.5,
                    },
                },
                customdata=[[species_text]],
                hovertemplate=(
                    "<b>Neue Beobachtung</b><br>"
                    f"{FEATURE_LABELS[x_feature]}: %{{x}}<br>"
                    f"{FEATURE_LABELS[y_feature]}: %{{y}}<br>"
                    "Vorhersage: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template="plotly_white",
        autosize=True,
        font={"family": "Arial, sans-serif", "color": "#30363b", "size": 11},
        legend={"orientation": "h", "x": 0, "y": 1.1, "title_text": ""},
        uirevision=f"{x_feature}:{y_feature}",
        margin={
            "l": 52,
            "r": 16,
            "t": 43,
            "b": 46,
        },
    )

    figure.update_xaxes(automargin=True, gridcolor="#ebedef", zeroline=False,
                        showline=True, linecolor="#bfc5ca", ticks="outside", tickcolor="#bfc5ca")
    figure.update_yaxes(automargin=True, gridcolor="#ebedef", zeroline=False,
                        showline=True, linecolor="#bfc5ca", ticks="outside", tickcolor="#bfc5ca")

    return figure
