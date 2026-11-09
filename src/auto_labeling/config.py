from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """All deterministic choices used by the experiment."""

    test_size: float = 0.20
    seeds: tuple[int, ...] = tuple(range(101, 111))
    support_fractions: tuple[float, ...] = (0.05, 0.10)
    n_clusters: int = 2
    positive_label: int = 0  # sklearn Breast Cancer: 0 = malignant
    output_dir: Path = Path("outputs")
    report_path: Path = Path("reports/informe_resultados.md")
    kmeans_n_init: int = 20
    viability_label_accuracy: float = 0.90
    viability_gap: float = 0.05
    algorithms: tuple[str, ...] = ("kmeans", "hierarchical", "gmm")
    classifiers: tuple[str, ...] = ("logistic_regression", "svm_rbf")
    first_seed_only_figures: bool = True

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be strictly between zero and one")
        if self.n_clusters != 2:
            raise ValueError("This study is designed for the binary k=2 setting")
        if not self.seeds:
            raise ValueError("At least one seed is required")
        if any(not 0 < fraction < 1 for fraction in self.support_fractions):
            raise ValueError("Support fractions must be strictly between zero and one")
