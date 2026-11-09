"""Run the complete clustering-assisted labelling study."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_labeling import ExperimentConfig, run_experiment


if __name__ == "__main__":
    results = run_experiment(ExperimentConfig())
    print("Experiment completed.")
    print(results["viability"][["algorithm", "support_fraction", "classifier", "viable"]].to_string(index=False))
