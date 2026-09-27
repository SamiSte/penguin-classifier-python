"""Schütze Datentrennung und vorab festgelegte Auswahl der Modellstudie."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from scripts import evaluate_regularization as study
from src.data_processing import load_clean_data, TARGET


def test_learning_samples_are_nested_stratified_and_keep_the_whole_last_fold():
    data = load_clean_data()
    protocol = study.make_protocol(data)
    held_out = set(protocol["held_out_indices"])
    for fold in protocol["cv_splits"]:
        subsets = study.learning_subsets(data, fold["train_indices"],
                                         study.LEARNING_FRACTIONS, 42 + fold["fold"])
        previous = set()
        forbidden = held_out | set(fold["validation_indices"])
        for fraction, subset in zip(study.LEARNING_FRACTIONS, subsets):
            actual = set(subset)
            assert previous <= actual
            assert not actual & forbidden
            assert set(data.iloc[subset][TARGET]) == set(data[TARGET])
            full_counts = data.iloc[fold["train_indices"]][TARGET].value_counts()
            counts = data.iloc[subset][TARGET].value_counts()
            assert ((counts - full_counts * fraction).abs() <= 1).all()
            previous = actual
        assert subsets[-1].tolist() == fold["train_indices"]


def test_every_variant_uses_the_same_folds_and_never_the_held_out_test(monkeypatch):
    data = load_clean_data()
    original_train, original_test = train_test_split(
        np.arange(len(data)), test_size=0.25, random_state=42, stratify=data[TARGET],
    )
    calls = []

    def observe(_data, train, validation, **params):
        assert not set(train) & set(validation)
        assert not (set(train) | set(validation)) & set(original_test)
        calls.append((tuple(train), tuple(validation), params))
        return dict(n_train=len(train), n_validation=len(validation),
                    train_accuracy=1.0, validation_accuracy=0.98,
                    train_macro_f1=1.0, validation_macro_f1=0.98,
                    tree_nodes=21.0, tree_depth=5.0, tree_leaves=11.0)

    monkeypatch.setattr(study, "score_fold", observe)
    result = study.run_study(data, progress=lambda _: None)
    assert result["protocol"]["training_indices"] == original_train.tolist()
    assert result["protocol"]["held_out_indices"] == original_test.tolist()
    assert len(result["variants"]) == 9
    assert len(calls) == 65  # 45 Varianten-Fits + 20 kleinere Lernkurven-Fits.
    first_folds = [(train, val) for train, val, _ in calls[:5]]
    for variant_number in range(9):
        variant_calls = calls[5 * variant_number:5 * (variant_number + 1)]
        assert [(train, val) for train, val, _ in variant_calls] == first_folds
    assert len(result["learning_curve"]) == 25
    assert sorted(i for _, validation in first_folds for i in validation) == sorted(original_train)


def test_selection_prefers_small_trees_only_within_declared_score_tolerance():
    def candidate(name, score, nodes):
        return {"name": name, "summary": {"validation_macro_f1_mean": score,
                                           "tree_nodes_mean": nodes}}
    variants = [candidate("best", 0.99, 100), candidate("compact", 0.986, 40),
                candidate("too_weak", 0.98, 10)]
    decision = study.choose_variant(variants)
    assert decision["best_score_variant"] == "best"
    assert decision["recommended_variant"] == "compact"
    assert decision["eligible_variants"] == ["best", "compact"]
    assert study.choose_variant(variants, tolerance=0)["recommended_variant"] == "best"


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_invalid_learning_fraction_is_rejected(fraction):
    data = load_clean_data()
    with pytest.raises(ValueError):
        study.learning_subsets(data, np.arange(len(data)), [fraction], 42)
