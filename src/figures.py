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
        title="Neue Beobachtung im Vergleich zu den Trainingsdaten",
    )

    figure.update_traces(
        marker={
            "size": 9,
            "opacity": 0.70,
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
        height=600,
        legend_title_text="Pinguinart",
        margin={
            "l": 60,
            "r": 30,
            "t": 80,
            "b": 60,
        },
    )

    return figure