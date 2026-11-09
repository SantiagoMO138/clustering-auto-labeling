from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_scores: np.ndarray,
    positive_label: int,
) -> dict[str, float]:
    """Return metrics with malignant encoded as the positive clinical class."""
    negative_label = 1 if positive_label == 0 else 0
    matrix = confusion_matrix(y_true, y_pred, labels=[positive_label, negative_label])
    true_negative = matrix[1, 1]
    false_positive = matrix[1, 0]
    specificity = true_negative / (true_negative + false_positive)
    y_binary = (np.asarray(y_true) == positive_label).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "malignant_sensitivity": float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_binary, positive_scores)),
    }


def clustering_external_metrics(cluster_labels: np.ndarray, y_reference: np.ndarray, accuracy: float) -> dict[str, float]:
    return {
        "cluster_accuracy_optimal": float(accuracy),
        "ari": float(adjusted_rand_score(y_reference, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(y_reference, cluster_labels)),
    }
