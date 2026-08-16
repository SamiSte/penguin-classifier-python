"""Trainiere, bewerte und speichere den Pinguinklassifikator."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data_processing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_clean_data,
)
from src.modeling import RANDOM_STATE, build_model_pipeline  # noqa: E402



MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "penguin_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def calculate_feature_ranges(data):
    """Berechne Wertebereiche und Median der numerischen Merkmale."""

    feature_ranges = {}

    for feature in NUMERIC_FEATURES:
        feature_ranges[feature] = {
            "minimum": float(data[feature].min()),
            "maximum": float(data[feature].max()),
            "median": float(data[feature].median()),
        }

    return feature_ranges


def calculate_cv_summary(cv_results):
    """Fasse die Ergebnisse der Kreuzvalidierung zusammen."""

    summary = {}

    for metric_name in ["accuracy", "f1_macro", "kappa"]:
        values = cv_results[f"test_{metric_name}"]

        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "individual_scores": [
                float(value) for value in values
            ],
        }

    return summary


def main() -> None:
    """Trainiere, bewerte und speichere das Modell samt Metadaten."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_clean_data()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n--- Datensatzaufteilung ---")
    print(f"Gesamtdaten: {len(data)}")
    print(f"Trainingsdaten: {len(X_train)}")
    print(f"Testdaten: {len(X_test)}")

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "kappa": make_scorer(cohen_kappa_score),
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\n--- Fünffache Kreuzvalidierung auf den Trainingsdaten ---")

    cv_results = cross_validate(
        estimator=build_model_pipeline(),
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cross_validation,
        n_jobs=1,
    )

    cv_summary = calculate_cv_summary(cv_results)

    for metric_name, result in cv_summary.items():
        print(
            f"{metric_name}: "
            f"{result['mean']:.4f} "
            f"(± {result['standard_deviation']:.4f})"
        )

    evaluation_model = build_model_pipeline()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )
    kappa = cohen_kappa_score(y_test, predictions)

    classes = evaluation_model.classes_.tolist()

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=classes,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    print("\n--- Leistung auf dem unabhängigen Testdatensatz ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    print("\n--- Klassenreihenfolge der Konfusionsmatrix ---")
    print(classes)

    print("\n--- Konfusionsmatrix ---")
    print(confusion)

    print("\n--- Klassenweiser Bericht ---")
    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,
            zero_division=0,
        )
    )
    
    print("\n--- Permutationswichtigkeit auf dem Testdatensatz ---")

    permutation_result = permutation_importance(
        estimator=evaluation_model,
        X=X_test,
        y=y_test,
        scoring="f1_macro",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    permutation_importances = []

    for feature, mean_importance, std_importance in zip(
        FEATURES,
        permutation_result.importances_mean,
        permutation_result.importances_std,
    ):
        permutation_importances.append(
            {
                "feature": feature,
                "mean_decrease_macro_f1": float(mean_importance),
                "standard_deviation": float(std_importance),
            }
        )

    permutation_importances.sort(
        key=lambda result: result["mean_decrease_macro_f1"],
        reverse=True,
    )

    for result in permutation_importances:
        print(
            f"{result['feature']}: "
            f"{result['mean_decrease_macro_f1']:.4f} "
            f"(± {result['standard_deviation']:.4f})"
        )

    metrics = {
        "data": {
            "total_observations": int(len(data)),
            "training_observations": int(len(X_train)),
            "test_observations": int(len(X_test)),
            "test_fraction": 0.25,
            "random_state": RANDOM_STATE,
        },
        "test_metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "cohen_kappa": float(kappa),
        },
        "permutation_importance": {
            "scoring": "f1_macro",
            "n_repeats": 30,
            "results": permutation_importances,
        },
        "cross_validation": cv_summary,
        "classes": classes,
        "confusion_matrix": confusion.tolist(),
        "classification_report": report,
    }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_ranges": calculate_feature_ranges(data),
        "allowed_values": {
            "sex": [
                "female",
                "male",
                "unknown",
            ],
            "island": sorted(
                data["island"].dropna().unique().tolist()
            ),
        },
        "classes": sorted(data[TARGET].unique().tolist()),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            ensure_ascii=False,
            indent=2,
        )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    # Das ausgelieferte Modell wird nach der unabhängigen Bewertung
    # auf allen 342 verfügbaren Beobachtungen trainiert.
    deployment_model = build_model_pipeline()
    deployment_model.fit(X, y)

    joblib.dump(
        deployment_model,
        MODEL_PATH,
    )

    print("\n--- Gespeicherte Dateien ---")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(METADATA_PATH)


if __name__ == "__main__":
    main()