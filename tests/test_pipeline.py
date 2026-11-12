import numpy as np

from auto_labeling.config import ExperimentConfig
from auto_labeling.pipeline import fit_clusterer, load_data, run_experiment, stratified_support_indices


def test_dataset_and_support_split_have_expected_shapes():
    X, y, _ = load_data()
    assert X.shape == (569, 30)
    assert set(np.unique(y)) == {0, 1}
    support = stratified_support_indices(y[:400], 0.10, 101)
    assert len(support) == 40


def test_clusterer_does_not_accept_labels_and_returns_valid_clusters():
    X, _, _ = load_data()
    config = ExperimentConfig(seeds=(101,))
    labels, _ = fit_clusterer("kmeans", X[:100], 101, config)
    assert len(labels) == 100
    assert set(np.unique(labels)).issubset({0, 1})


def test_small_experiment_is_reproducible_and_metric_ranges_are_valid(tmp_path):
    config = ExperimentConfig(seeds=(101, 102), output_dir=tmp_path / "one", report_path=tmp_path / "one.md")
    first = run_experiment(config, create_artifacts=False)
    second = run_experiment(config, create_artifacts=False)
    assert first["cluster_results"].equals(second["cluster_results"])
    assert first["classifier_results"].equals(second["classifier_results"])
    metric_columns = ["accuracy", "malignant_sensitivity", "specificity", "precision", "f1", "roc_auc"]
    assert ((first["classifier_results"][metric_columns] >= 0) & (first["classifier_results"][metric_columns] <= 1)).all().all()


def test_support_real_baseline_is_recorded_once_per_seed_fraction_and_classifier(tmp_path):
    config = ExperimentConfig(seeds=(101, 102), output_dir=tmp_path / "one", report_path=tmp_path / "one.md")
    results = run_experiment(config, create_artifacts=False)
    support_rows = results["classifier_results"].query("scheme == 'support_real'")
    expected = len(config.seeds) * len(config.support_fractions) * len(config.classifiers)
    assert len(support_rows) == expected
    assert support_rows.groupby(["seed", "support_fraction", "classifier"]).size().eq(1).all()
