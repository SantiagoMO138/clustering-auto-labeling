from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import ExperimentConfig


ALGORITHM_LABELS = {
    "kmeans": "K-Means",
    "hierarchical": "Jerárquico (Ward)",
    "gmm": "GMM",
    "full_real": "Etiquetas reales completas",
    "support_real": "Etiquetas reales de apoyo",
}


def write_artifacts(result: dict, visual_data: dict | None, config: ExperimentConfig, feature_names: list[str]) -> None:
    """Persist raw data, summaries, figures and a concise academic report."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cluster_results", "classifier_results", "cluster_summary", "classifier_summary", "viability", "ranking"):
        result[name].to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(result["config"]), stream, indent=2, ensure_ascii=False)

    if visual_data:
        _plot_pca_clusters(visual_data, output_dir / "pca_clusters.png")
    _plot_generated_accuracy(result["ranking"], output_dir / "generated_accuracy.png")
    _plot_gaps(result["viability"], output_dir / "performance_gaps.png")
    _write_report(result, config, feature_names)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _plot_pca_clusters(visual_data: dict, destination: Path) -> None:
    X_2d = PCA(n_components=2, random_state=int(visual_data["seed"])).fit_transform(visual_data["X_train"])
    clusters = visual_data["clusters"]
    columns = len(clusters) + 1
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 4), constrained_layout=True)
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=visual_data["y_train"], cmap="coolwarm", s=16)
    axes[0].set_title("Clase real oculta\n0=maligno, 1=benigno")
    for axis, (algorithm, labels) in zip(axes[1:], clusters.items(), strict=True):
        axis.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=16)
        axis.set_title(ALGORITHM_LABELS[algorithm])
    for axis in axes:
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
    figure.suptitle(f"Visualización PCA - entrenamiento (semilla {visual_data['seed']})")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_generated_accuracy(ranking: pd.DataFrame, destination: Path) -> None:
    frame = ranking.copy()
    frame["label"] = frame.apply(lambda row: f"{ALGORITHM_LABELS[row['algorithm']]}\n{int(row['support_fraction'] * 100)}% - {row['classifier']}", axis=1)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(frame["label"], frame["accuracy_mean"], yerr=frame["accuracy_ci95"], capsize=4, color="#386cb0")
    axis.set_ylabel("Exactitud media en prueba")
    axis.set_ylim(0, 1.05)
    axis.tick_params(axis="x", labelrotation=45)
    axis.set_title("Desempeño de clasificadores con pseudoetiquetas")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_gaps(viability: pd.DataFrame, destination: Path) -> None:
    frame = viability.copy()
    frame["label"] = frame.apply(lambda row: f"{ALGORITHM_LABELS[row['algorithm']]}\n{int(row['support_fraction'] * 100)}% - {row['classifier']}", axis=1)
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.37
    axis.bar(positions - width / 2, frame["accuracy_gap"], width, label="Brecha de exactitud")
    axis.bar(positions + width / 2, frame["sensitivity_gap"], width, label="Brecha de sensibilidad")
    axis.axhline(0.05, color="crimson", linestyle="--", label="Umbral absoluto: 5 pp")
    axis.axhline(-0.05, color="crimson", linestyle="--")
    axis.set_xticks(positions, frame["label"], rotation=45, ha="right")
    axis.set_ylabel("Diferencia frente a etiquetas completas")
    axis.legend()
    axis.set_title("Brechas de desempeño de pseudoetiquetas")
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(result: dict, config: ExperimentConfig, feature_names: list[str]) -> None:
    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    viability = result["viability"]
    ranking = result["ranking"]
    viable_count = int(viability["viable"].sum())
    top = ranking.iloc[0]
    lines = [
        "# Informe de resultados: etiquetado automático mediante clustering",
        "",
        "## Objetivo y diseño",
        "",
        "Se evaluó la generación de pseudoetiquetas con K-Means, clustering jerárquico Ward y Gaussian Mixture Models (GMM) en Breast Cancer Wisconsin (Diagnostic). El dataset contiene 569 observaciones y 30 variables morfológicas. La clase `0` (maligno) se trató como positiva.",
        "",
        f"Se realizaron {len(config.seeds)} particiones estratificadas entrenamiento/prueba de {int((1-config.test_size)*100)}/{int(config.test_size*100)}. En cada entrenamiento se estandarizaron las variables sin usar datos de prueba y se simuló apoyo experto estratificado del 5 % y 10 %. Las etiquetas de diagnóstico no se entregaron a los algoritmos de clustering.",
        "",
        "## Criterio de viabilidad",
        "",
        "Un enfoque se considera viable si la exactitud media de etiquetado con la correspondencia obtenida del apoyo experto es al menos 90 %, y las brechas absolutas de exactitud y sensibilidad maligna frente al clasificador entrenado con etiquetas completas no exceden 5 puntos porcentuales.",
        "",
        f"Se identificaron **{viable_count}** combinaciones viables de {len(viability)} evaluadas.",
        "",
        "## Mejor resultado por desempeño promedio",
        "",
        f"La combinación líder fue **{ALGORITHM_LABELS[top['algorithm']]}**, apoyo del **{int(top['support_fraction'] * 100)} %** y **{top['classifier']}**, con exactitud media de **{top['accuracy_mean']:.3f} +/- {top['accuracy_ci95']:.3f}** y sensibilidad maligna de **{top['malignant_sensitivity_mean']:.3f} +/- {top['malignant_sensitivity_ci95']:.3f}**.",
        "",
        "## Resultados detallados",
        "",
        "Los resultados crudos, resúmenes, ranking y evaluación de viabilidad se encuentran en `outputs/`. Las figuras principales son `pca_clusters.png`, `generated_accuracy.png` y `performance_gaps.png`.",
        "",
        "## Limitaciones",
        "",
        "Este resultado corresponde a un dataset clínico curado y de tamaño reducido; no sustituye diagnóstico clínico ni justifica extrapolación a otros hospitales, poblaciones o problemas de clasificación.",
        "",
        f"Variables utilizadas: {', '.join(feature_names)}.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
