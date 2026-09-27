"""Gemeinsame Vorverarbeitung und begrenzter Random Forest."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


RANDOM_STATE = 42
N_ESTIMATORS = 500
# Die Modellstudie erlaubt kleinere Bäume bei höchstens 0,5 Prozentpunkten CV-Verlust.
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 1


def build_model_pipeline() -> Pipeline:
    """Fehlendes Geschlecht ergänzen, Kategorien kodieren und Forest aufbauen."""
    categories = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", categories, CATEGORICAL_FEATURES),
    ], remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
