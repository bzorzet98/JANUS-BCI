"""
`MLflowLogger` -- centraliza TODO el logueo a MLflow (hyperparámetros,
resumen de dataset, métricas de test), separado de TrainingOrchestrator.

Por qué existe: TrainingOrchestrator empezó a acumular varios métodos
_log_* (hyperparams, dataset summary, métricas) que no tienen relación
con "orquestar una corrida de training" -- son responsabilidad de
"cómo se reporta a MLflow", una preocupación distinta. Separarlas en su
propio colaborador evita que el orquestador siga creciendo con lógica de
reporting, y deja un solo lugar para el manejo de tipos frágil
(tensor/numpy) que antes estaba disperso.

MLflowLogger NO abre ni cierra runs de MLflow -- eso sigue siendo
responsabilidad del orquestador (Camino B, ya decidido en sesión: el
orquestador es el único que tiene el config completo y decide cuándo
empieza/termina una corrida trackeada). MLflowLogger solo loguea DENTRO
de un run ya abierto, igual que MLflowCallback.
"""
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import torch 

def to_numpy(x) -> np.ndarray:
    """Único lugar del proyecto que debe resolver "esto puede ser tensor
    o numpy, no sé cuál" -- centralizado acá porque el manejo disperso de
    `hasattr(y, 'cpu')` es frágil (silenciosamente incorrecto si alguien
    pasa un tipo raro que tiene .cpu() pero no es tensor de torch, por
    ejemplo). Si aparece la misma necesidad en otro módulo que NO sea de
    tracking/reporting, mover esto a un lugar más genérico
    (training/utils.py) en vez de duplicarlo ahí."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


class MLflowLogger:
    def __init__(self, metrics_strategy):
        self.metrics_strategy = metrics_strategy

    def log_hyperparams(self, config: dict) -> None:
        """Solo model/optimizer/criterion params -- identidad (project,
        strategy, partition, label) ya está en tags del run, no se
        duplica acá."""
        params = {}
        for block_name in ("model", "optimizer", "criterion"):
            block_params = config.get(block_name, {}).get("params", {})
            for key, value in block_params.items():
                params[f"{block_name}.{key}"] = value
        mlflow.log_params(params)

    def log_dataset_summary(self, fold, dataset_dir_name: str, label: str) -> None:
        """Dataset real de MLflow (mlflow.data) -- resume shape y balance
        de clases por split, para poder inspeccionar la partición sin
        salir de la UI ni abrir el CSV de outputs."""
        summary_rows = []
        splits = [("train", fold.X_train, fold.y_train), ("test", fold.X_test, fold.y_test)]
        if fold.X_val is not None:
            splits.insert(1, ("val", fold.X_val, fold.y_val))

        for split_name, X, y in splits:
            X_np, y_np = to_numpy(X), to_numpy(y)
            class_counts = pd.Series(y_np).value_counts().to_dict()
            summary_rows.append({
                "split": split_name,
                "n_trials": X_np.shape[0],
                "n_channels": X_np.shape[1],
                "n_samples": X_np.shape[2],
                **{f"n_class_{cls}": count for cls, count in class_counts.items()},
            })

        summary_df = pd.DataFrame(summary_rows)
        dataset_info = mlflow.data.from_pandas(
            summary_df, source=dataset_dir_name, name=f"{dataset_dir_name}_{label}",
        )
        mlflow.log_input(dataset_info, context="training")

    def log_test_metrics(self, fold, y_pred_test, partition: str, seed: int, append_csv_row_fn, metrics_csv_path) -> None:
        """Calcula métricas de test vía MetricsStrategy, y las loguea
        TANTO a metrics_results.csv (vía append_csv_row_fn, inyectada
        para no acoplar este logger a run_lifecycle directamente) COMO a
        MLflow -- un solo lugar, evita la duplicación que había entre
        _train_dl_replicate y _train_ml_replicate."""
        y_test_np = to_numpy(fold.y_test)
        y_pred_test_np = to_numpy(y_pred_test)
        y_test_pred_class = y_pred_test_np.argmax(axis=1)

        metrics_rows = self.metrics_strategy.compute(
            y_test_np, y_test_pred_class, y_pred_proba=y_pred_test_np, split="test",
        )
        for row in metrics_rows:
            row["partition"] = partition
            row["model_init_seed"] = seed
            append_csv_row_fn(metrics_csv_path, row)
            mlflow.log_metric(row["metric_name"], row["value"])

    def log_duration(self, history: dict, append_csv_row_fn, metrics_csv_path, partition: str, seed: int) -> None:
        """duration es un escalar (no una serie por época, a diferencia
        de train_loss/val_loss) -- va como fila de metrics_results.csv y
        como métrica de MLflow, no al CSV de historia por época."""
        if "duration" not in history:
            return
        append_csv_row_fn(metrics_csv_path, {
            "split": "train", "metric_name": "duration_seconds", "value": history["duration"],
            "partition": partition, "model_init_seed": seed,
        })
        mlflow.log_metric("duration_seconds", history["duration"])