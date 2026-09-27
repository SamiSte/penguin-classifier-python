"""Prüfe Datenintegrität und unabhängige Bewertung der Modellaktualisierung."""

import csv
import json

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import modeling, retraining_data
from src.data_processing import DATA_PATH, FEATURES, NUMERIC_FEATURES, TARGET
from src.retraining_data import prepare_dataset, train_and_evaluate
from src.storage import ALLOWED_SPECIES, CSV_COLUMNS


@pytest.fixture
def reference_path(tmp_path):
    """Drei ausreichend große Klassen mit eindeutig identifizierbaren Merkmalen."""
    rows = []
    for class_number, species in enumerate(ALLOWED_SPECIES):
        for number in range(12):
            rows.append(
                {
                    "species": species,
                    "island": "Biscoe",
                    "bill_length_mm": 35 + class_number * 10 + number / 10,
                    "bill_depth_mm": 15 + class_number + number / 10,
                    "flipper_length_mm": 180 + class_number * 20 + number,
                    "body_mass_g": 3000 + class_number * 1000 + number * 10,
                    "sex": "female" if number % 2 else "male",
                }
            )
    path = tmp_path / "reference.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_observations(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def new_observation(**changes):
    return {
        "bill_length_mm": 50.5,
        "bill_depth_mm": 18,
        "flipper_length_mm": 201,
        "body_mass_g": 4300,
        "sex": "",
        "island": "Dream",
        "predicted_species": "Adelie",
        "validated_species": "Chinstrap",
        **changes,
    }


def test_real_reference_preserves_all_342_clean_cases_and_unknown_sex(tmp_path):
    data, report = prepare_dataset(DATA_PATH, tmp_path / "absent.csv")
    assert len(data) == 342
    assert data["sex"].eq("unknown").sum() == 9
    assert data["source"].eq("reference").all()
    assert report["confirmed_count"] == 0
    assert report["confirmed_ids"] == []
    assert report["confirmed_class_counts"] == dict.fromkeys(ALLOWED_SPECIES, 0)
    assert sum(report["class_counts"].values()) == 342
    assert not (tmp_path / "absent.csv").exists()


def test_only_independent_confirmation_becomes_label(reference_path, tmp_path):
    observations = write_observations(
        tmp_path / "observations.csv",
        [
            new_observation(),
            new_observation(validated_species="", bill_length_mm="invalid"),
        ],
    )
    original = observations.read_bytes()
    data, report = prepare_dataset(reference_path, observations)
    confirmed = data.loc[data["source"].eq("confirmed")]
    assert confirmed[TARGET].tolist() == ["Chinstrap"]
    assert confirmed["sex"].tolist() == ["unknown"]
    assert "predicted_species" not in data.columns
    assert report["confirmed_count"] == 1
    assert report["ignored_unconfirmed"] == 1
    assert report["confirmed_class_counts"] == {"Adelie": 0, "Chinstrap": 1, "Gentoo": 0}
    assert report["warnings"]
    assert observations.read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("bill_length_mm", ""),
        ("bill_length_mm", "text"),
        ("bill_length_mm", "NaN"),
        ("bill_depth_mm", "inf"),
        ("flipper_length_mm", 0),
        ("body_mass_g", -100),
        ("sex", "other"),
        ("validated_species", "Emperor"),
    ],
)
def test_invalid_confirmed_row_has_location_and_never_changes_source(
    reference_path, tmp_path, field, value
):
    observations = write_observations(
        tmp_path / "observations.csv", [new_observation(**{field: value})]
    )
    before = observations.read_bytes()
    with pytest.raises(ValueError, match="Zeile 2"):
        prepare_dataset(reference_path, observations)
    assert observations.read_bytes() == before


@pytest.mark.parametrize("malformed", ["", "species,bill_length_mm\nAdelie,44\n"])
def test_missing_schema_is_not_mistaken_for_no_confirmations(reference_path, tmp_path, malformed):
    observations = tmp_path / "observations.csv"
    observations.write_text(malformed, encoding="utf-8")
    with pytest.raises(ValueError, match="Zeile 1.*Spalten"):
        prepare_dataset(reference_path, observations)


@pytest.mark.parametrize("malformed_row", ['"unterminated', "1,2,3"])
def test_malformed_csv_is_rejected_with_location(reference_path, tmp_path, malformed_row):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        ",".join(CSV_COLUMNS) + "\n" + malformed_row, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Zeile 2"):
        prepare_dataset(reference_path, observations)


def test_context_does_not_make_duplicates_new_training_data(reference_path, tmp_path):
    reference_first = pd.read_csv(reference_path).iloc[0].to_dict()
    duplicate_reference = {
        key: value for key, value in reference_first.items() if key in CSV_COLUMNS
    }
    duplicate_reference.update(validated_species=reference_first[TARGET], island="Dream")
    observations = write_observations(
        tmp_path / "observations.csv",
        [
            duplicate_reference,
            new_observation(),
            new_observation(island="Torgersen", sex="unknown", bill_depth_mm="18.00"),
        ],
    )
    data, report = prepare_dataset(reference_path, observations)
    assert len(data) == 37
    assert report["duplicates_removed"] == 2
    assert report["confirmed_count"] == 1
    assert data["source"].eq("reference").sum() == 36
    assert data["feature_id"].is_unique


@pytest.mark.parametrize("conflicts_with_reference", [False, True])
def test_conflicting_labels_for_identical_features_stop_training(
    reference_path, tmp_path, conflicts_with_reference
):
    if conflicts_with_reference:
        base = pd.read_csv(reference_path).iloc[0].to_dict()
        rows = [
            {
                **{key: value for key, value in base.items() if key in CSV_COLUMNS},
                "validated_species": "Gentoo",
            }
        ]
        line_number = 2
    else:
        rows = [new_observation(), new_observation(validated_species="Gentoo")]
        line_number = 3
    observations = write_observations(tmp_path / "observations.csv", rows)
    with pytest.raises(ValueError, match=f"Zeile {line_number}.*Widersprüchliche"):
        prepare_dataset(reference_path, observations)


def test_identities_are_stable_across_context_numeric_spelling_and_row_order(reference_path, tmp_path):
    observations = write_observations(tmp_path / "observations.csv", [new_observation()])
    first, first_report = prepare_dataset(reference_path, observations)
    pd.read_csv(reference_path).iloc[::-1].to_csv(reference_path, index=False)
    write_observations(
        observations,
        [new_observation(body_mass_g="4300.00", island="Torgersen", sex="unknown")],
    )
    second, second_report = prepare_dataset(reference_path, observations)
    assert first_report["reference_hash"] == second_report["reference_hash"]
    assert first_report["confirmed_ids"] == second_report["confirmed_ids"]
    assert set(first["record_id"]) == set(second["record_id"])
    assert all(len(value) == 64 for value in first["record_id"])


def test_csv_read_returns_a_complete_snapshot_before_later_appends(tmp_path):
    path = write_observations(tmp_path / "observations.csv", [new_observation()])
    snapshot = retraining_data._confirmed_rows(path)
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=CSV_COLUMNS).writerow(new_observation())
    assert len(snapshot) == 1
    assert len(retraining_data._confirmed_rows(path)) == 2


@pytest.fixture
def fast_forest(monkeypatch):
    monkeypatch.setattr(modeling, "N_ESTIMATORS", 3)


def test_all_evaluation_fits_exclude_validation_and_deployment_uses_all_data(
    reference_path, tmp_path, fast_forest, monkeypatch
):
    baseline, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    observations = write_observations(tmp_path / "observations.csv", [new_observation()])
    candidate, _ = prepare_dataset(reference_path, observations)
    # Even an accidental confirmed copy of a validation observation must not
    # reintroduce that feature combination into either evaluation training set.
    reference = baseline.sort_values("record_id").reset_index(drop=True)
    _, validation = train_test_split(
        reference, test_size=0.25, random_state=42, stratify=reference[TARGET]
    )
    alias = validation.iloc[[0]].copy()
    alias["source"] = "confirmed"
    candidate = pd.concat([candidate, alias], ignore_index=True)
    baseline = pd.concat([baseline, alias], ignore_index=True)
    fits = []
    original_fit = Pipeline.fit

    def record_fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame) and list(X.columns) == FEATURES:
            fits.append(set(X.itertuples(index=False, name=None)))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    progress = []
    result = train_and_evaluate(candidate, baseline, progress=progress.append)
    validation_features = set(validation[FEATURES].itertuples(index=False, name=None))
    candidate_features = set(candidate[FEATURES].itertuples(index=False, name=None))
    assert len(fits) == 8  # two comparisons, five CV folds, deployment
    assert all(not fitted.intersection(validation_features) for fitted in fits[:-1])
    assert fits[-1] == candidate_features
    metrics = result["metrics"]
    assert metrics["validation_size"] == 9
    assert metrics["baseline_training_size"] == 27
    assert metrics["candidate_training_size"] == 28
    assert set(metrics["validation_feature_ids"]) == set(validation["feature_id"])
    assert metrics["classes"] == list(ALLOWED_SPECIES)
    for name in ("baseline", "candidate"):
        assert 0 <= metrics[name]["accuracy"] <= 1
        assert 0 <= metrics[name]["macro_f1"] <= 1
        assert -1 <= metrics[name]["cohen_kappa"] <= 1
        assert sum(map(sum, metrics[name]["confusion_matrix"])) == 9
    assert set(metrics["cross_validation"]) == {"accuracy", "macro_f1", "cohen_kappa"}
    assert all(
        len(scores["individual_scores"]) == 5
        for scores in metrics["cross_validation"].values()
    )
    assert result["model"].named_steps["classifier"].class_weight is None
    assert result["metadata"]["features"] == FEATURES
    assert set(result["metadata"]["feature_ranges"]) == set(NUMERIC_FEATURES)
    assert len(progress) == 4
    json.dumps(metrics, allow_nan=False)
    json.dumps(result["metadata"], allow_nan=False)


def test_validation_remains_the_same_after_reference_reordering_and_more_confirmations(
    reference_path, tmp_path, fast_forest
):
    baseline, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    first = train_and_evaluate(baseline, baseline)
    observations = write_observations(tmp_path / "observations.csv", [new_observation()])
    pd.read_csv(reference_path).iloc[::-1].to_csv(reference_path, index=False)
    candidate, _ = prepare_dataset(reference_path, observations)
    second = train_and_evaluate(candidate, baseline)
    assert first["metrics"]["validation_feature_ids"] == second["metrics"]["validation_feature_ids"]
    assert first["metrics"]["baseline"] == second["metrics"]["baseline"]


def test_changed_snapshot_features_fail_before_training(reference_path, tmp_path):
    baseline, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    candidate = baseline.copy()
    candidate.loc[0, "bill_length_mm"] += 1
    with pytest.raises(ValueError, match="Trainingsdaten wurden verändert"):
        train_and_evaluate(candidate, baseline)


def test_changed_reference_is_not_compared_with_old_baseline(reference_path, tmp_path):
    baseline, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    changed = pd.read_csv(reference_path)
    changed.loc[0, "bill_length_mm"] += 1
    changed.to_csv(reference_path, index=False)
    candidate, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="Referenzdaten unterscheiden"):
        train_and_evaluate(candidate, baseline)


def test_comparison_clones_previous_configuration_without_reusing_fitted_trees(
    reference_path, tmp_path, fast_forest, monkeypatch
):
    data, _ = prepare_dataset(reference_path, tmp_path / "missing.csv")
    previous = modeling.build_model_pipeline().set_params(classifier__max_depth=None)
    previous.fit(data[FEATURES], data[TARGET])
    original_tree = previous.named_steps["classifier"].estimators_[0]
    fits = []
    real_fit = Pipeline.fit

    def record_fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame) and list(X.columns) == FEATURES:
            fits.append((self.named_steps["classifier"].max_depth, len(X)))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    result = train_and_evaluate(data, data, baseline_pipeline=previous)
    assert fits[0] == (None, 27)
    assert all(depth == 5 for depth, _ in fits[1:])
    assert previous.named_steps["classifier"].estimators_[0] is original_tree
    assert result["metrics"]["baseline_parameters"]["max_depth"] is None
    assert result["metrics"]["candidate_parameters"]["max_depth"] == 5
