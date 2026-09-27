"""Separate Modellstudie auf dem Trainingsanteil; verändert keine App-Modelle.

Aufruf: python scripts/evaluate_regularization.py
Ergebnisse: docs/modellpruefung/ (JSON, CSV, Markdown und offlinefähiges HTML).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import DATA_PATH, FEATURES, TARGET, load_clean_data  # noqa: E402
from src.modeling import N_ESTIMATORS, RANDOM_STATE, build_model_pipeline  # noqa: E402

CV_FOLDS = 5
SELECTION_TOLERANCE = 0.005
LEARNING_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "modellpruefung"


def make_protocol(data):
    """Reproduziere Ersttrainingssplit; CV-Indizes beziehen sich auf Gesamtdaten."""
    train, held_out = train_test_split(
        np.arange(len(data)), test_size=0.25, random_state=RANDOM_STATE,
        stratify=data[TARGET],
    )
    folds = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for number, (fit, validate) in enumerate(cv.split(train, data.iloc[train][TARGET]), 1):
        folds.append({"fold": number, "train_indices": train[fit].tolist(),
                      "validation_indices": train[validate].tolist()})
    return {
        "seed": RANDOM_STATE, "cv_folds": CV_FOLDS,
        "total_count": len(data), "training_count": len(train),
        "held_out_count": len(held_out), "training_indices": train.tolist(),
        "held_out_indices": held_out.tolist(), "cv_splits": folds,
        "selection_tolerance": SELECTION_TOLERANCE,
        "n_estimators": N_ESTIMATORS,
        "learning_fractions": list(LEARNING_FRACTIONS),
    }


def learning_subsets(data, train_indices, fractions, seed):
    """Verschachtelte, annähernd stratifizierte Teilmengen innerhalb eines Folds.

Je Klasse wird einmal gemischt und pro Größe ein wachsender Anteil entnommen.
Rundungen können die tatsächliche Gesamtgröße um wenige Fälle verschieben.
Die letzte Teilmenge enthält alle Trainingsfälle in der ursprünglichen Reihenfolge.
"""
    train_indices = np.asarray(train_indices)
    labels = data.iloc[train_indices][TARGET].to_numpy()
    rng = np.random.default_rng(seed)
    members = [rng.permutation(train_indices[labels == name]) for name in np.unique(labels)]
    subsets = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Lernkurvenanteile müssen zwischen 0 (exklusiv) und 1 liegen.")
        selected = set(np.concatenate([
            indices[:max(1, round(fraction * len(indices)))] for indices in members
        ]).tolist())
        subsets.append(np.array([i for i in train_indices if i in selected], dtype=int))
    return subsets


def score_fold(data, train_indices, validation_indices, *, max_depth, min_samples_leaf):
    """Fitte auch die Vorverarbeitung ausschließlich auf den jeweiligen Trainingsfällen."""
    model = build_model_pipeline().set_params(
        classifier__max_depth=max_depth, classifier__min_samples_leaf=min_samples_leaf,
    )
    training = data.iloc[train_indices]
    validation = data.iloc[validation_indices]
    model.fit(training[FEATURES], training[TARGET])
    result = {"n_train": len(training), "n_validation": len(validation)}
    for prefix, subset in (("train", training), ("validation", validation)):
        predictions = model.predict(subset[FEATURES])
        result[f"{prefix}_accuracy"] = float(accuracy_score(subset[TARGET], predictions))
        result[f"{prefix}_macro_f1"] = float(f1_score(
            subset[TARGET], predictions, average="macro", zero_division=0,
        ))
    trees = model.named_steps["classifier"].estimators_
    result.update(
        tree_nodes=float(np.mean([tree.tree_.node_count for tree in trees])),
        tree_depth=float(np.mean([tree.get_depth() for tree in trees])),
        tree_leaves=float(np.mean([tree.get_n_leaves() for tree in trees])),
    )
    return result


def summarize(folds):
    summary = {}
    for metric in ("train_accuracy", "validation_accuracy", "train_macro_f1", "validation_macro_f1"):
        values = [fold[metric] for fold in folds]
        summary[f"{metric}_mean"] = float(np.mean(values))
        # Deskriptive Streuung der fünf vorhandenen Folds, kein Konfidenzintervall.
        summary[f"{metric}_std"] = float(np.std(values, ddof=0))
    for metric in ("tree_nodes", "tree_depth", "tree_leaves"):
        summary[f"{metric}_mean"] = float(np.mean([fold[metric] for fold in folds]))
    return summary


def choose_variant(variants, tolerance=SELECTION_TOLERANCE):
    """Vorab definierte Regel: ähnliche CV-Güte, dann kleinere mittlere Bäume."""
    best = max(variants, key=lambda item: item["summary"]["validation_macro_f1_mean"])
    cutoff = best["summary"]["validation_macro_f1_mean"] - tolerance
    eligible = [item for item in variants
                if item["summary"]["validation_macro_f1_mean"] >= cutoff - 1e-12]
    selected = min(eligible, key=lambda item: (
        item["summary"]["tree_nodes_mean"],
        -item["summary"]["validation_macro_f1_mean"],
        item["name"],
    ))
    return {"best_score_variant": best["name"], "recommended_variant": selected["name"],
            "eligible_variants": [item["name"] for item in eligible], "tolerance": tolerance}


def run_study(data, progress=print):
    """Vergleiche neun Varianten und erstelle eine Lernkurve für die Baseline."""
    protocol = make_protocol(data)
    variants = []
    for depth in (None, 5, 10):
        for leaf in (1, 2, 4):
            name = f"depth_{'none' if depth is None else depth}_leaf_{leaf}"
            progress(f"Variante {len(variants) + 1}/9: max_depth={depth}, min_samples_leaf={leaf}")
            folds = []
            for split in protocol["cv_splits"]:
                result = score_fold(data, split["train_indices"], split["validation_indices"],
                                    max_depth=depth, min_samples_leaf=leaf)
                folds.append({"fold": split["fold"], **result})
            variants.append({"name": name, "max_depth": depth, "min_samples_leaf": leaf,
                             "folds": folds, "summary": summarize(folds)})

    baseline = variants[0]
    curve = []
    for split, full_result in zip(protocol["cv_splits"], baseline["folds"]):
        progress(f"Lernkurve: Fold {split['fold']}/{CV_FOLDS}")
        subsets = learning_subsets(data, split["train_indices"], LEARNING_FRACTIONS,
                                   RANDOM_STATE + split["fold"])
        for fraction, subset in zip(LEARNING_FRACTIONS, subsets):
            result = full_result if fraction == 1.0 else score_fold(
                data, subset, split["validation_indices"], max_depth=None, min_samples_leaf=1,
            )
            curve.append({"fraction": fraction, "fold": split["fold"],
                          "train_indices": subset.tolist(), **result})
    return {"schema_version": 1, "protocol": protocol, "variants": variants,
            "learning_curve": curve, "selection": choose_variant(variants)}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for protected in (PROJECT_ROOT / "models", PROJECT_ROOT / "data"):
        if output == protected.resolve() or output.is_relative_to(protected.resolve()):
            parser.error("Studienergebnisse dürfen nicht im Modell- oder Datenordner liegen.")
    study = run_study(load_clean_data(), progress=lambda value: print(value, flush=True))
    study["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "data_sha256": file_hash(DATA_PATH), "script_sha256": file_hash(__file__),
        "modeling_sha256": file_hash(PROJECT_ROOT / "src" / "modeling.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ergebnisse.json").write_text(
        json.dumps(study, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    from scripts.model_study_report import write_report
    write_report(study, output)
    print(f"Empfohlene Variante: {study['selection']['recommended_variant']}", flush=True)
    print(f"Ergebnisse: {output}", flush=True)


if __name__ == "__main__":
    main()
