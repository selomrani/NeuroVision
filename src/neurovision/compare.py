"""Print the MLflow runs side by side.

    python -m neurovision.compare
"""

from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = {
    "tags.mlflow.runName": "run",
    "params.total_params": "params",
    "params.learning_rate": "lr",
    "metrics.val_accuracy": "val_acc",
    "metrics.val_f1_macro": "val_f1",
    "metrics.duration_minutes": "minutes",
}


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["NeuroVision"])
    if runs.empty:
        print("no runs yet, start with: python -m neurovision.train")
        return

    table = runs[list(COLUMNS)].rename(columns=COLUMNS)
    table = table.sort_values("val_f1", ascending=False).round(3)
    print(table.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
