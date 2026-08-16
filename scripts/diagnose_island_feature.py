"""Diagnostischer Vergleich des Modells mit und ohne Inselmerkmal."""

from pathlib import Path
import sys

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing import (  # noqa: E402
    NUMERIC_FEATURES,
    TARGET,
    load_clean_data,
)
from src.modeling import RANDOM_STATE, N_ESTIMATORS  # noqa: E402


FEATURES_WITH_ISLAND = [
    *NUMERIC_FEATURES,
    "sex",
    "island",
]

FEATURES_WITHOUT_ISLAND = [
    *NUMERIC_FEATURES,
    "sex",
]


def build_pipeline(categorical_features: list[str]) -> Pipeline:
    """Erstelle eine Pipeline für die übergebenen kategorialen Merkmale."""

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
                OneHotEncoder(handle_unknown="ignore"),
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
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_variant(
    name: str,
    features: list[str],
    categorical_features: list[str],
    train_indices,
    test_indices,
    data,
) -> None:
    """Bewerte eine Merkmalsvariante auf identischer Datenaufteilung."""

    X = data[features]
    y = data[TARGET]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    model = build_pipeline(categorical_features)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )
    kappa = cohen_kappa_score(y_test, predictions)

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = cross_validate(
        estimator=build_pipeline(categorical_features),
        X=X_train,
        y=y_train,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        },
        cv=cross_validation,
        n_jobs=1,
    )

    print(f"\n--- {name} ---")
    print(f"Merkmale: {features}")
    print(f"Test-Accuracy: {accuracy:.4f}")
    print(f"Test-Macro-F1: {macro_f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(
        "CV-Accuracy: "
        f"{np.mean(cv_results['test_accuracy']):.4f} "
        f"(± {np.std(cv_results['test_accuracy']):.4f})"
    )
    print(
        "CV-Macro-F1: "
        f"{np.mean(cv_results['test_f1_macro']):.4f} "
        f"(± {np.std(cv_results['test_f1_macro']):.4f})"
    )


def main() -> None:
    """Vergleiche die Modellleistung mit und ohne Insel."""

    data = load_clean_data()
    y = data[TARGET]

    all_indices = np.arange(len(data))

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    evaluate_variant(
        name="Modell mit Insel",
        features=FEATURES_WITH_ISLAND,
        categorical_features=["sex", "island"],
        train_indices=train_indices,
        test_indices=test_indices,
        data=data,
    )

    evaluate_variant(
        name="Modell ohne Insel",
        features=FEATURES_WITHOUT_ISLAND,
        categorical_features=["sex"],
        train_indices=train_indices,
        test_indices=test_indices,
        data=data,
    )


if __name__ == "__main__":
    main()
