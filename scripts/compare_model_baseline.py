"""Vergleiche den kompakten Forest mit einer einfachen logistischen Regression.

Nur die bisherigen Trainingsdaten gehen in die identischen CV-Folds ein.
Gespeicherte Modelle und Beobachtungen werden nicht verändert.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_regularization import make_protocol
from src.data_processing import DATA_PATH, FEATURES, TARGET, load_clean_data
from src.modeling import build_model_pipeline


def build_variant(kind):
    pipeline = build_model_pipeline()
    if kind == "logistic":
        pipeline.set_params(preprocessor__numeric=StandardScaler(),
                            classifier=LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
    else:
        pipeline.set_params(classifier__max_depth=5 if kind == "compact" else None,
                            classifier__min_samples_leaf=1)
    return pipeline


def compare(data, progress=print):
    protocol = make_protocol(data)
    results = []
    for kind, label in (("original", "Random Forest ohne Tiefenbegrenzung"),
                        ("compact", "Random Forest mit maximaler Tiefe 5"),
                        ("logistic", "Logistische Regression mit L2-Regularisierung")):
        progress(label)
        folds = []
        for split in protocol["cv_splits"]:
            training = data.iloc[split["train_indices"]]
            validation = data.iloc[split["validation_indices"]]
            model = build_variant(kind)
            start = perf_counter()
            model.fit(training[FEATURES], training[TARGET])
            seconds = perf_counter() - start
            predicted = model.predict(validation[FEATURES])
            trained = model.predict(training[FEATURES])
            fold = {
                "fold": split["fold"], "fit_seconds": seconds,
                "accuracy": float(accuracy_score(validation[TARGET], predicted)),
                "macro_f1": float(f1_score(validation[TARGET], predicted, average="macro")),
                "cohen_kappa": float(cohen_kappa_score(validation[TARGET], predicted)),
                "training_macro_f1": float(f1_score(training[TARGET], trained, average="macro")),
            }
            classifier = model.named_steps["classifier"]
            if kind != "logistic":
                fold.update(leaves=float(np.mean([t.get_n_leaves() for t in classifier.estimators_])),
                            nodes=float(np.mean([t.tree_.node_count for t in classifier.estimators_])))
            else:
                fold["coefficients"] = int(classifier.coef_.size + classifier.intercept_.size)
            folds.append(fold)
        summary = {key: {"mean": float(np.mean([f[key] for f in folds])),
                         "std": float(np.std([f[key] for f in folds]))}
                   for key in folds[0] if key != "fold"}
        results.append({"kind": kind, "label": label, "folds": folds, "summary": summary})
    return {"protocol": protocol, "results": results}


def main():
    report = compare(load_clean_data(), progress=lambda text: print(text, flush=True))
    report["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = ROOT / "docs" / "modellpruefung"
    output.mkdir(parents=True, exist_ok=True)
    (output / "einfaches_modell.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Vergleich mit einem einfacheren Modell", "",
             "Identische fünf stratifizierte CV-Folds auf 256 Trainingsbeobachtungen. "
             "Die 86 ursprünglichen Testbeobachtungen werden nicht verwendet. "
             "Die Standardisierung der logistischen Regression wird pro Trainingsfold gelernt. "
             "C = 1 und die L2-Regularisierung bleiben unverändert; es erfolgt keine Parametersuche.", "",
             "| Modell | CV Accuracy | CV Macro-F1 | CV Kappa | Blätter je Baum |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for model in report["results"]:
        values = model["summary"]
        scores = [f"{values[key]['mean']:.4f} ± {values[key]['std']:.4f}".replace(".", ",")
                  for key in ("accuracy", "macro_f1", "cohen_kappa")]
        leaves = f"{values['leaves']['mean']:.1f}".replace(".", ",") if "leaves" in values else "–"
        lines.append(f"| {model['label']} | {' | '.join(scores)} | {leaves} |")
    lines += ["", "Die Streuung beschreibt die fünf Folds und ist kein Konfidenzintervall. "
              "Die gleiche Kreuzvalidierung wurde bereits für die Wahl der Baumtiefe verwendet. "
              "Der Vergleich ist daher explorativ und kein neuer unabhängiger Gütenachweis. "
              "Mehr künftige Beobachtungen allein begründen keine Überlegenheit des Random Forest. "
              "Die einfachere Alternative sollte bei späteren Datenständen erneut mitgeführt werden.", "",
              "Vollständige Einzelwerte und Aufteilungen: [einfaches_modell.json](einfaches_modell.json).", "",
              "Methodischer Bezug: [scikit-learn RandomForestClassifier]"
              "(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) "
              "und [logistische Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)."]
    (output / "einfaches_modell.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps([{r["kind"]: r["summary"]} for r in report["results"]], indent=2))


if __name__ == "__main__":
    main()
