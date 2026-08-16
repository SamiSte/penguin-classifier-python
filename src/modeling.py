"""Aufbau der Modellpipeline für die Pinguinklassifikation."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


RANDOM_STATE = 42
N_ESTIMATORS = 500


def build_model_pipeline() -> Pipeline:
    """Erstelle die vollständige Vorverarbeitungs- und Modellpipeline.

    Numerische Merkmale werden unverändert an den Random Forest
    weitergegeben. Fehlende kategoriale Werte werden durch ``unknown``
    ersetzt und anschließend per One-Hot-Encoding umgewandelt.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Nicht trainierte Pipeline aus Vorverarbeitung und Klassifikator.
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="unknown",
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model_pipeline
