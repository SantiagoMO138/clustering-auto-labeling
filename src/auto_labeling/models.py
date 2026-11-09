from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .metrics import classification_metrics


def build_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    if name == "svm_rbf":
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    raise ValueError(f"Unsupported classifier: {name}")


def fit_and_score_classifier(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    positive_label: int,
    seed: int,
) -> dict[str, float]:
    model = build_classifier(name, seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    positive_index = int(np.where(model.classes_ == positive_label)[0][0])
    positive_scores = model.predict_proba(X_test)[:, positive_index]
    return classification_metrics(y_test, predictions, positive_scores, positive_label)
