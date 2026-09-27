"""Der Vergleich einfacher Modelle darf keine Prüfdaten zum Lernen verwenden."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts import compare_model_baseline as study
from src.data_processing import FEATURES, NUMERIC_FEATURES, load_clean_data


def test_identical_folds_and_fold_local_scaling_exclude_original_test_data(monkeypatch):
    data = load_clean_data()
    protocol = study.make_protocol(data)
    held_out = set(data.iloc[protocol["held_out_indices"]].index)
    build = study.build_variant

    def small_variant(kind):
        model = build(kind)
        if kind != "logistic":
            model.set_params(classifier__n_estimators=3, classifier__n_jobs=1)
        return model

    monkeypatch.setattr(study, "build_variant", small_variant)
    fit = Pipeline.fit
    fitted_indices = []
    scaling_checks = []

    def record_fit(self, X, y=None, **kwargs):
        result = fit(self, X, y, **kwargs)
        if isinstance(X, pd.DataFrame) and list(X.columns) == FEATURES:
            fitted_indices.append(tuple(X.index))
            scaler = self.named_steps["preprocessor"].named_transformers_["numeric"]
            if isinstance(scaler, StandardScaler):
                scaling_checks.append(np.allclose(scaler.mean_, X[NUMERIC_FEATURES].mean()))
        return result

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    study.compare(data, progress=lambda _: None)
    assert len(fitted_indices) == 15
    assert all(not held_out.intersection(indices) for indices in fitted_indices)
    assert fitted_indices[:5] == fitted_indices[5:10] == fitted_indices[10:]
    assert scaling_checks == [True] * 5
