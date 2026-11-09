from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .labeling import apply_cluster_mapping, map_clusters_to_classes, optimal_label_accuracy
from .metrics import classification_metrics, clustering_external_metrics
from .models import fit_and_score_classifier
from .reporting import write_artifacts


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the canonical 569x30 diagnostic dataset from scikit-learn."""
    dataset = load_breast_cancer()
    X = dataset.data.astype(float)
    y = dataset.target.astype(int)  # 0 = malignant; 1 = benign
    if X.shape != (569, 30):
        raise ValueError(f"Unexpected dataset shape: {X.shape}")
    return X, y, list(dataset.feature_names)


def fit_clusterer(name: str, X_train: np.ndarray, seed: int, config: ExperimentConfig) -> tuple[np.ndarray, object]:
    """Fit clustering without accepting or accessing diagnostic labels."""
    if name == "kmeans":
        model = KMeans(n_clusters=config.n_clusters, n_init=config.kmeans_n_init, random_state=seed)
        labels = model.fit_predict(X_train)
    elif name == "hierarchical":
        model = AgglomerativeClustering(n_clusters=config.n_clusters, linkage="ward")
        labels = model.fit_predict(X_train)
    elif name == "gmm":
        model = GaussianMixture(n_components=config.n_clusters, covariance_type="full", random_state=seed)
        labels = model.fit_predict(X_train)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {name}")
    return np.asarray(labels, dtype=int), model


def stratified_support_indices(y_train: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return the labelled expert-support subset, stratified by diagnosis."""
    indices = np.arange(len(y_train))
    support, _ = train_test_split(indices, train_size=fraction, stratify=y_train, random_state=seed)
    return np.sort(support)


def _base_rows(seed: int, algorithm: str, fraction: float, cluster_labels: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, mapped_labels: np.ndarray) -> dict[str, float | int | str]:
    cluster_accuracy = optimal_label_accuracy(cluster_labels, y_train)
    external = clustering_external_metrics(cluster_labels, y_train, cluster_accuracy)
    return {
        "seed": seed,
        "algorithm": algorithm,
        "support_fraction": fraction,
        "silhouette": float(silhouette_score(X_train, cluster_labels)),
        "davies_bouldin": float(davies_bouldin_score(X_train, cluster_labels)),
        **external,
        "labeling_accuracy_support_mapping": float(np.mean(mapped_labels == y_train)),
    }


def run_experiment(config: ExperimentConfig | None = None, create_artifacts: bool = True) -> dict[str, pd.DataFrame | dict]:
    """Run all repetitions and return raw and aggregated experimental results."""
    config = config or ExperimentConfig()
    config.validate()
    X, y, feature_names = load_data()
    cluster_rows: list[dict] = []
    classifier_rows: list[dict] = []
    visual_data: dict[str, object] | None = None

    for seed in config.seeds:
        train_indices, test_indices = train_test_split(
            np.arange(len(y)), test_size=config.test_size, stratify=y, random_state=seed
        )
        X_train_raw, X_test_raw = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for classifier_name in config.classifiers:
            baseline_metrics = fit_and_score_classifier(
                classifier_name, X_train, y_train, X_test, y_test, config.positive_label, seed
            )
            classifier_rows.append({
                "seed": seed, "algorithm": "full_real", "support_fraction": 1.0,
                "classifier": classifier_name, "scheme": "full_real", **baseline_metrics,
            })

        clusters_by_algorithm: dict[str, np.ndarray] = {}
        for algorithm in config.algorithms:
            cluster_labels, _ = fit_clusterer(algorithm, X_train, seed, config)
            clusters_by_algorithm[algorithm] = cluster_labels
            for fraction in config.support_fractions:
                support_indices = stratified_support_indices(y_train, fraction, seed)
                mapping = map_clusters_to_classes(cluster_labels[support_indices], y_train[support_indices])
                pseudo_labels = apply_cluster_mapping(cluster_labels, mapping)
                cluster_rows.append(_base_rows(seed, algorithm, fraction, cluster_labels, X_train, y_train, pseudo_labels))

                for classifier_name in config.classifiers:
                    generated_metrics = fit_and_score_classifier(
                        classifier_name, X_train, pseudo_labels, X_test, y_test, config.positive_label, seed
                    )
                    classifier_rows.append({
                        "seed": seed, "algorithm": algorithm, "support_fraction": fraction,
                        "classifier": classifier_name, "scheme": "generated", **generated_metrics,
                    })
                    support_metrics = fit_and_score_classifier(
                        classifier_name, X_train[support_indices], y_train[support_indices], X_test, y_test,
                        config.positive_label, seed,
                    )
                    classifier_rows.append({
                        "seed": seed, "algorithm": "support_real", "support_fraction": fraction,
                        "classifier": classifier_name, "scheme": "support_real", **support_metrics,
                    })

        if visual_data is None or not config.first_seed_only_figures:
            visual_data = {
                "seed": seed,
                "X_train": X_train,
                "y_train": y_train,
                "clusters": clusters_by_algorithm.copy(),
            }

    cluster_results = pd.DataFrame(cluster_rows)
    classifier_results = pd.DataFrame(classifier_rows)
    summary = summarize_results(cluster_results, classifier_results, config)
    result: dict[str, pd.DataFrame | dict] = {
        "cluster_results": cluster_results,
        "classifier_results": classifier_results,
        "cluster_summary": summary["cluster_summary"],
        "classifier_summary": summary["classifier_summary"],
        "viability": summary["viability"],
        "ranking": summary["ranking"],
        "config": asdict(config),
    }
    if create_artifacts:
        write_artifacts(result, visual_data, config, feature_names)
    return result


def summarize_results(cluster_results: pd.DataFrame, classifier_results: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    metrics = ["silhouette", "davies_bouldin", "cluster_accuracy_optimal", "ari", "nmi", "labeling_accuracy_support_mapping"]
    cluster_summary = _aggregate(cluster_results, ["algorithm", "support_fraction"], metrics)
    classifier_metrics = ["accuracy", "malignant_sensitivity", "specificity", "precision", "f1", "roc_auc"]
    classifier_summary = _aggregate(classifier_results, ["algorithm", "support_fraction", "classifier", "scheme"], classifier_metrics)

    generated = classifier_summary[classifier_summary["scheme"] == "generated"].copy()
    full = classifier_summary[classifier_summary["scheme"] == "full_real"][["classifier", "accuracy_mean", "malignant_sensitivity_mean"]].copy()
    full = full.rename(columns={"accuracy_mean": "full_accuracy_mean", "malignant_sensitivity_mean": "full_malignant_sensitivity_mean"})
    viability = cluster_summary.merge(generated[["algorithm", "support_fraction", "classifier", "accuracy_mean", "malignant_sensitivity_mean"]], on=["algorithm", "support_fraction"])
    viability = viability.merge(full, on="classifier")
    viability["accuracy_gap"] = viability["full_accuracy_mean"] - viability["accuracy_mean"]
    viability["sensitivity_gap"] = viability["full_malignant_sensitivity_mean"] - viability["malignant_sensitivity_mean"]
    viability["viable"] = (
        (viability["labeling_accuracy_support_mapping_mean"] >= config.viability_label_accuracy)
        & (viability["accuracy_gap"].abs() <= config.viability_gap)
        & (viability["sensitivity_gap"].abs() <= config.viability_gap)
    )
    ranking = generated.sort_values(["accuracy_mean", "malignant_sensitivity_mean"], ascending=False).reset_index(drop=True)
    return {"cluster_summary": cluster_summary, "classifier_summary": classifier_summary, "viability": viability, "ranking": ranking}


def _aggregate(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=1).fillna(0).add_suffix("_std")
    counts = grouped.size().rename("n")
    result = pd.concat([means, stds, counts], axis=1).reset_index()
    for metric in metric_columns:
        result[f"{metric}_ci95"] = 1.96 * result[f"{metric}_std"] / np.sqrt(result["n"])
    return result
