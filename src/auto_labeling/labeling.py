from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def map_clusters_to_classes(
    support_clusters: np.ndarray,
    support_labels: np.ndarray,
    cluster_ids: tuple[int, ...] = (0, 1),
    class_ids: tuple[int, ...] = (0, 1),
) -> dict[int, int]:
    """Build a one-to-one cluster-to-class map using only expert support labels.

    The Hungarian assignment maximises observed support agreement.  Supplying
    the complete identifiers, rather than only observed values, makes the
    deterministic fallback valid even if a small support subset misses a
    cluster.
    """
    support_clusters = np.asarray(support_clusters)
    support_labels = np.asarray(support_labels)
    if support_clusters.shape != support_labels.shape:
        raise ValueError("support_clusters and support_labels must have equal shapes")

    counts = np.zeros((len(cluster_ids), len(class_ids)), dtype=int)
    cluster_index = {value: index for index, value in enumerate(cluster_ids)}
    class_index = {value: index for index, value in enumerate(class_ids)}
    for cluster, label in zip(support_clusters, support_labels, strict=True):
        if cluster in cluster_index and label in class_index:
            counts[cluster_index[int(cluster)], class_index[int(label)]] += 1

    row_indices, col_indices = linear_sum_assignment(-counts)
    return {cluster_ids[row]: class_ids[col] for row, col in zip(row_indices, col_indices, strict=True)}


def apply_cluster_mapping(cluster_labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Convert arbitrary cluster identifiers to binary class labels."""
    labels = np.asarray(cluster_labels)
    try:
        return np.asarray([mapping[int(cluster)] for cluster in labels], dtype=int)
    except KeyError as error:
        raise ValueError(f"No class mapping exists for cluster {error.args[0]}") from error


def optimal_label_accuracy(cluster_labels: np.ndarray, reference_labels: np.ndarray) -> float:
    """External clustering accuracy using Hungarian matching for evaluation only."""
    mapping = map_clusters_to_classes(cluster_labels, reference_labels)
    return float(np.mean(apply_cluster_mapping(cluster_labels, mapping) == reference_labels))
