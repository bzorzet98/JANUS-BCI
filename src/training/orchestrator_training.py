"""
TrainingOrchestrator -- clase que coordina un barrido completo de
entrenamiento: RunLifecycleManager, trío de reproducibilidad, MLflow, y
el loop de unidades -> Schema.generate_folds() -> Trainer_DL/Trainer_ML.

Completamente agnóstico de qué Schema se usa: siempre llama
schema.units_to_process() para saber qué recorrer, y
schema.generate_folds(unit_id=...) con la misma keyword unificada para
los 3 Schemas -- nunca pregunta "¿qué Schema es este?".

El Generator de DataLoaders se arma DESDE fold.metadata["dataloader_seed"],
nunca creado una sola vez para todo el barrido -- cada Schema decide su
propia semántica de reseteo, el orquestador solo usa lo que recibe.

Nota: CallbackDispatcher NO se instancia acá -- Trainer_DL ya lo compone
internamente. El orquestador solo arma la LISTA de callbacks (vía
build_callbacks + los defaults de _setup_default_callbacks) y se la pasa
a Trainer_DL.

Nota de organización de métodos: los _setup_* son métodos de INSTANCIA,
no @staticmethod, aunque varios solo lean self.config hoy -- pertenecen
conceptualmente al ciclo de vida de ESTA corrida, no son utilidades
genéricas reusables fuera de un TrainingOrchestrator en ejecución.
resolve_model_init_seeds, en cambio, es una función SUELTA en
training/utils.py -- transformación pura config->lista, reusable tal cual
desde HPSearchOrchestrator u otro contexto.

Nota de esta pasada: todo el logueo a MLflow (hyperparams, dataset
summary, métricas de test, duración) se extrajo a MLflowLogger
(training/tracking/mlflow_logger.py) -- separa "orquestar el training" de
"reportar a MLflow", que son responsabilidades distintas. También se
unificó la cola final común entre _train_dl_replicate/_train_ml_replicate
(write outputs + log métricas) en _finalize_replicate, eliminando la
duplicación literal que había entre ambos métodos. to_numpy (manejo
tensor/numpy frágil) quedó centralizado en mlflow_logger.py -- si aparece
la misma necesidad fuera de contexto de tracking, mover a
training/utils.py en vez de duplicar.

preload_to_device: Trainer_DL/Tester_DL deciden, cada uno con su propio
flag, si el batch se mueve a GPU de antemano o batch a batch dentro del
loop (default, escala mejor). create_dataset NUNCA mueve nada a GPU --
el movimiento es responsabilidad exclusiva de Trainer_DL.model_infer.

Config esperado (ver sesión de diseño):
  project_config: {project_name, strategy_name, recipe_name, database_name, database_session, label}
  script_type: "train_dl" | "train_ml"
  databases: {class_name, module_name, params}
  data_provider: {class_name, module_name, params}   -- OBLIGATORIO, sin default BCI
  schema: {class_name, module_name, params: {..., units_to_test, dataloader_seed, ...}}
  orchestrator: {model_init_seed}                     -- device se pasa por CLI/flag, no en config
  model, optimizer, criterion, callbacks: como ya existían
  torch_dataset, torch_dataloader: {train, train_eval, val, test}
  trainer: {params: {max_epochs, preload_to_device, ...}}
  metrics_strategy: {class_name, module_name, params} -- OPCIONAL, default ClassificationMetrics()
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import torch

from src.training.core.callbacks import LossTracker, TimerCallback
from src.training.core.seeds import set_seed
from src.training.core.trainer import Trainer_DL, Trainer_ML
from src.training.data_provider import DataProvider
from src.training.metrics.classification import ClassificationMetrics
from src.training.splitting.schemas import SCHEMA_REGISTRY
from src.training.tracking.callbacks_mlflow import MLflowCallback
from src.training.tracking.mlflow_logger import MLflowLogger, to_numpy
from src.training.utils import build_callbacks, create_dataloader, create_dataset, resolve_model_init_seeds
from src.utils.imports import import_class
from src.utils.json_utils import save_dict_as_json
from src.utils.paths import PATHS
from src.utils.progress_reporter import ProgressReporter
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, _append_csv_row, _clear_sweep_outputs, _drop_csv_rows

logger = logging.getLogger(__name__)

TRAINER_REGISTRY = {
    "train_dl": Trainer_DL,
    "train_ml": Trainer_ML,
}


class TrainingOrchestrator:
    """Coordina un barrido completo de training. Instanciar una vez por
    corrida (un config), llamar .run()."""

    def __init__(
        self, config: dict, docker_image: Optional[str] = None, device: str = "cpu",
        reporter: Optional[ProgressReporter] = None, force: bool = False, overwrite_existing: bool = False,
    ):
        self.config = config
        self.docker_image = docker_image
        self.device = device
        self.reporter = reporter
        self.force = force
        self.overwrite_existing = overwrite_existing

        project_config = config["project_config"]
        self.project_name = project_config["project_name"]
        self.strategy_name = project_config["strategy_name"]
        self.recipe_name = project_config["recipe_name"]
        self.database_name = project_config["database_name"]
        self.database_session = project_config["database_session"]
        self.label = project_config["label"]
        self.script_type = config.get("script_type", "train_dl")

        self.dataset_dir_name = f"{self.database_name}_{self.database_session}"
        self.output_path = (
            PATHS.results_root / self.project_name / self.strategy_name / self.recipe_name /
            self.dataset_dir_name / self.label
        )

        self.provider: Optional[DataProvider] = None
        self.schema = None
        self.lifecycle: Optional[RunLifecycleManager] = None
        self.metrics_strategy = None
        self.mlflow_logger: Optional[MLflowLogger] = None

    # --- setup, cada uno una responsabilidad chica y nombrada ---
    # Métodos de INSTANCIA (no @staticmethod) -- ver nota de organización
    # en el docstring del módulo.

    def _setup_provider(self) -> DataProvider:
        """Instancia el dataset real vía import_class (config['databases']),
        lo envuelve en el adapter configurado en config['data_provider'].
        NO hay default BCI-específico: 'data_provider' es OBLIGATORIO."""
        dataset_cfg = self.config["databases"]
        dataset = import_class(dataset_cfg["class_name"], dataset_cfg["module_name"])(**dataset_cfg["params"])

        provider_cfg = self.config.get("data_provider")
        if provider_cfg is None:
            raise ValueError(
                "config['data_provider'] es obligatorio -- no hay default agnóstico de dominio."
            )

        provider_class = import_class(provider_cfg["class_name"], provider_cfg["module_name"])
        params = dict(provider_cfg.get("params", {}))
        params["dataset"] = dataset
        return provider_class(**params)

    def _setup_schema(self, provider: DataProvider):
        """Instancia el Schema vía import_class desde config['schema'].
        split_strategy y label_transform, si están en el config, también
        se resuelven vía import_class -- si no están, el Schema usa sus
        propios defaults."""
        schema_cfg = self.config["schema"]
        schema_class = SCHEMA_REGISTRY.get(schema_cfg["class_name"])
        if schema_class is None:
            schema_class = import_class(schema_cfg["class_name"], schema_cfg["module_name"])

        params = dict(schema_cfg.get("params", {}))
        params["provider"] = provider

        if "split_strategy" in schema_cfg:
            ss_cfg = schema_cfg["split_strategy"]
            params["split_strategy"] = import_class(ss_cfg["class_name"], ss_cfg["module_name"])(**ss_cfg.get("params", {}))
        if "label_transform" in schema_cfg:
            lt_cfg = schema_cfg["label_transform"]
            params["label_transform"] = import_class(lt_cfg["class_name"], lt_cfg["module_name"])(**lt_cfg.get("params", {}))

        return schema_class(**params)

    def _setup_lifecycle(self) -> RunLifecycleManager:
        lifecycle = RunLifecycleManager(
            self.output_path,
            force=self.force, overwrite_existing=self.overwrite_existing,
            emit_running_row=True,
        )
        lifecycle.check_identity_or_raise(
            new_config=self.config,
            extra_cleanup=lambda: _clear_sweep_outputs(self.output_path),
            context_label=f"la corrida de training en {self.output_path}",
        )
        log_reproducibility_trio(self.output_path, self.config, self.docker_image)
        return lifecycle

    def _setup_pending_units(self) -> Dict[Any, List[str]]:
        """Devuelve {unit_id: [particiones pendientes de esa unidad]}, en
        el orden de units_to_process()."""
        units = self.schema.units_to_process()

        # Expandir unidades a particiones esperadas -- el lifecycle trabaja
        # en términos de partición (nombre de carpeta), no de unidad lógica
        # (un unit_id puede generar varias particiones, ej. K folds).
        unit_by_partition = {}
        for unit_id in units:
            for partition in self.schema.expected_partition_names(unit_id):
                unit_by_partition[partition] = unit_id

        def _still_present(partition: str) -> bool:
            return (self.output_path / partition).exists()

        pending_partitions = self.lifecycle.prepare_partitions(list(unit_by_partition.keys()), still_present=_still_present)

        if not pending_partitions:
            logger.info("Todo ya procesado (%d particiones) -- nada para hacer.", len(unit_by_partition))
            return {}

        # Agrupado por unidad: generate_folds se llama una vez por unidad
        # (carga sus datos una sola vez), pero _run_unit solo entrena los
        # folds cuya partición sigue pendiente -- ej. si 3 de los 5 folds
        # de una unidad K-Fold ya están en success, no se reentrenan ni se
        # duplican sus filas en metrics_results.csv.
        pending_by_unit: Dict[Any, List[str]] = {}
        for partition in pending_partitions:
            pending_by_unit.setdefault(unit_by_partition[partition], []).append(partition)
        return pending_by_unit

    def _setup_metrics_strategy(self):
        """Resuelve MetricsStrategy desde config['metrics_strategy'], o
        ClassificationMetrics() default (solo accuracy) si no está."""
        metrics_cfg = self.config.get("metrics_strategy")
        if metrics_cfg is None:
            return ClassificationMetrics()
        metrics_class = import_class(metrics_cfg["class_name"], metrics_cfg["module_name"])
        return metrics_class(**metrics_cfg.get("params", {}))

    def _setup_model(self):
        model_cfg = self.config["model"]
        return import_class(model_cfg["class_name"], model_cfg["module_name"])(**model_cfg["params"])

    def _setup_optimizer(self, model):
        optimizer_cfg = self.config["optimizer"]
        return import_class(optimizer_cfg["class_name"], optimizer_cfg["module_name"])(model.parameters(), **optimizer_cfg["params"])

    def _setup_criterion(self):
        criterion_cfg = self.config["criterion"]
        return import_class(criterion_cfg["class_name"], criterion_cfg["module_name"])(**criterion_cfg["params"])

    def _setup_default_callbacks(self, has_val: bool) -> List:
        """LossTracker (train + val si corresponde) y TimerCallback SIEMPRE
        activos por default -- información base que siempre querés tener,
        costo marginal. MLflowCallback también siempre, tonto respecto a
        MLflow (el orquestador ya abrió el run)."""
        callbacks = [TimerCallback(), LossTracker(name="train_loss", on_training=True), MLflowCallback()]
        if has_val:
            callbacks.append(LossTracker(name="val_loss", on_training=False))
        return callbacks

    def _setup_single_loader(self, split_name: str, X, y, g: Optional[torch.Generator], preload_to_device: bool):
        """Arma UN DataLoader para cualquier split (train/train_eval/val/test),
        aplicando SIEMPRE el mismo criterio de preload_to_device -- reusado
        tanto por el armado de train/val (antes de entrenar) como por la
        inferencia post-training (train_eval/val/test), para que ningún
        camino se salte el movimiento a device."""
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        if preload_to_device:
            X = X.to(self.device)
            y = y.to(self.device)

        dataset = create_dataset(split_name, X, y, self.config["torch_dataset"])
        return create_dataloader(split_name, dataset, self.config["torch_dataloader"], g)

    def _setup_dl_dataloaders(self, fold, g: torch.Generator, preload_to_device: bool):
        """Solo train/val -- para armar el training loop."""
        train_loader = self._setup_single_loader("train", fold.X_train, fold.y_train, g, preload_to_device)

        val_loader = None
        if fold.X_val is not None:
            val_loader = self._setup_single_loader("val", fold.X_val, fold.y_val, g, preload_to_device)

        return train_loader, val_loader

    def _setup_dl_callbacks(self, weights_dir: Path, seed: int, optimizer, has_val: bool) -> List:
        """Solo arma la lista de callbacks -- default + los del config, con
        dirname/prefix/optimizer inyectados."""
        return build_callbacks(
            base_callbacks=self._setup_default_callbacks(has_val=has_val),
            config_callbacks=self.config.get("callbacks"),
            aditional_params={
                "path_to_save": str(weights_dir),
                "prefix": f"seed_{seed}_",
                "optimizer": optimizer,
            },
        )

    # --- ejecución ---
    def run(self) -> None:
        self.provider = self._setup_provider()
        self.schema = self._setup_schema(self.provider)
        self.metrics_strategy = self._setup_metrics_strategy()
        self.mlflow_logger = MLflowLogger(self.metrics_strategy)
        self.lifecycle = self._setup_lifecycle()
        pending_units = self._setup_pending_units()

        model_init_seeds = resolve_model_init_seeds(self.config["orchestrator"])

        mlflow.set_experiment(f"{self.project_name}/{self.strategy_name}")

        # El reporter avanza una unidad por partición (on_run_result), no
        # por unit_id -- con K-Fold una unidad aporta K particiones.
        n_pending_partitions = sum(len(partitions) for partitions in pending_units.values())

        if self.reporter:
            self.reporter.on_start(n_pending_partitions, meta={
                "strategy_name": self.strategy_name, "recipe_name": self.recipe_name,
                "database_name": self.database_name, "n_units": len(pending_units),
                "model_init_seeds": model_init_seeds,
            })

        for unit_id, partitions in pending_units.items():
            self._run_unit(unit_id, partitions, model_init_seeds)

        logger.info("--- Training Finished ---")
        if self.reporter:
            self.reporter.on_finish({"total": n_pending_partitions})

    def _run_unit(self, unit_id, pending_partitions: List[str], model_init_seeds: List[int]) -> None:
        """Una unidad puede producir MÁS DE UN Fold (ej. WithinUnitKFoldSchema
        con K folds) -- cada Fold es una partición independiente en disco,
        cada una recorre las N seeds de init de pesos.

        script_progress.csv se escribe POR PARTICIÓN (unit_01,
        unit_01_fold_0, loo_test-unit_01...), nunca por unit_id: es la
        misma clave que usan el resume (_setup_pending_units, vía
        expected_partition_names) y TrainedModelResolver en testing (primer
        segmento del path de los pesos).

        Un fallo nunca corta el barrido: se loguea con traceback completo
        (logger.exception), la partición queda 'failed' y el próximo
        barrido la reintenta."""
        try:
            folds = self.schema.generate_folds(unit_id=unit_id)
        except Exception:
            # Sin folds no hay a qué partición atribuirle el error más que
            # a todas las pendientes de esta unidad.
            logger.exception("Fallo generando los folds de la unidad %s", unit_id)
            for partition in pending_partitions:
                ts_start = self.lifecycle.mark_start(partition)
                if self.reporter:
                    self.reporter.on_run_start(partition)
                    self.reporter.on_run_result(partition, "failed")
                self.lifecycle.mark_result(partition, "failed", ts_start)
            return

        for fold in folds:
            partition = self.schema.partition_name(fold)
            if partition not in pending_partitions:
                continue

            ts_start = self.lifecycle.mark_start(partition)
            status = "success"
            if self.reporter:
                self.reporter.on_run_start(partition)

            # Reproceso (reintento tras 'failed', o --force): las filas que
            # dejó la pasada anterior de ESTA partición se descartan antes
            # de entrenar, en vez de duplicarse. Se reentrenan todas sus
            # seeds, no solo las que faltaban: g se comparte entre las seeds
            # del fold, así que saltear una cambiaría el orden de batches de
            # las siguientes respecto de una corrida sin cortes.
            _drop_csv_rows(self.output_path / "metrics_results.csv", "partition", partition)

            try:
                run_dir = self.output_path / partition
                run_dir.mkdir(parents=True, exist_ok=True)
                save_dict_as_json(str(run_dir), fold.metadata, "fold_metadata.json")

                # Generator armado desde lo que EL SCHEMA decidió -- fix
                # del hallazgo del Generator compartido entre unidades.
                g = torch.Generator()
                g.manual_seed(fold.metadata["dataloader_seed"])

                for seed in model_init_seeds:
                    self._run_replicate(fold, partition, seed, g)
            except Exception:
                logger.exception("Fallo procesando la partición %s", partition)
                status = "failed"

            if self.reporter:
                self.reporter.on_run_result(partition, status)
            self.lifecycle.mark_result(partition, status, ts_start)

    def _run_replicate(self, fold, partition: str, seed: int, g: torch.Generator) -> None:
        """Una réplica = un (fold, seed). set_seed afecta SOLO init de
        pesos -- g (shuffling de DataLoaders) es compartido entre las N
        seeds del MISMO fold, a propósito."""
        set_seed(seed)

        run_dir = self.output_path / partition
        model = self._setup_model()

        trainer_class = TRAINER_REGISTRY[self.script_type]

        tags = {
            "project_name": self.project_name,
            "strategy_name": self.strategy_name,
            "recipe_name": self.recipe_name,
            "database_name": self.database_name,
            "label": self.label,
            "partition": partition,
            "model_init_seed": str(seed),
        }

        with mlflow.start_run(run_name=f"{partition}_seed{seed}", tags=tags):
            self.mlflow_logger.log_hyperparams(self.config)
            self.mlflow_logger.log_dataset_summary(fold, self.dataset_dir_name, self.label)

            if trainer_class is Trainer_DL:
                y_pred_train, y_pred_val, y_pred_test, history = self._train_dl_replicate(fold, model, run_dir, seed, g)
            else:
                y_pred_train, y_pred_val, y_pred_test, history = self._train_ml_replicate(fold, model, run_dir, seed)

            self._finalize_replicate(fold, y_pred_train, y_pred_val, y_pred_test, history, run_dir, partition, seed)

    def _write_outputs_csv(self, fold, y_pred_train, y_pred_val, y_pred_test, run_dir: Path, seed: int) -> None:
        """Un CSV único por seed: trial_idx (absoluto, mismo espacio que
        fold.metadata['train_idx']/etc.), split, y_true, y_pred,
        proba_0..proba_N."""
        rows = []
        splits = [
            ("train", fold.metadata["train_idx"], fold.y_train, y_pred_train),
            ("test", fold.metadata["test_idx"], fold.y_test, y_pred_test),
        ]
        if fold.X_val is not None:
            splits.insert(1, ("val", fold.metadata["val_idx"], fold.y_val, y_pred_val))

        for split_name, idx_list, y_true, y_pred_proba in splits:
            y_true_np = to_numpy(y_true)
            y_pred_proba_np = to_numpy(y_pred_proba)
            y_pred_class = y_pred_proba_np.argmax(axis=1)
            n_classes = y_pred_proba_np.shape[1]
            for i, trial_idx in enumerate(idx_list):
                row = {
                    "trial_idx": trial_idx, "split": split_name,
                    "y_true": int(y_true_np[i]), "y_pred": int(y_pred_class[i]),
                }
                for cls in range(n_classes):
                    row[f"proba_{cls}"] = float(y_pred_proba_np[i, cls])
                rows.append(row)

        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outputs_dir / f"seed_{seed}_outputs.csv", index=False)

    def _write_history_csv(self, history: dict, run_dir: Path, seed: int) -> None:
        """CSV por seed: epoch, train_loss, val_loss -- valores crudos de
        trainer.history, para poder analizar curvas fuera de MLflow. NO se
        guarda training_plot.png -- decisión ya tomada, MLflow cubre la
        visualización."""
        n_epochs = len(history.get("train_loss", []))
        rows = []
        for epoch in range(n_epochs):
            row = {"epoch": epoch}
            for key in ("train_loss", "val_loss"):
                if key in history and epoch < len(history[key]):
                    row[key] = history[key][epoch]
            rows.append(row)

        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outputs_dir / f"seed_{seed}_history.csv", index=False)

    def _finalize_replicate(self, fold, y_pred_train, y_pred_val, y_pred_test, history: dict, run_dir: Path, partition: str, seed: int) -> None:
        """Cola COMÚN entre _train_dl_replicate y _train_ml_replicate --
        antes estaba duplicada literal en ambos métodos (write outputs +
        log métricas de test + log duración). Un solo lugar ahora."""
        self._write_outputs_csv(fold, y_pred_train, y_pred_val, y_pred_test, run_dir, seed)
        if history:
            self._write_history_csv(history, run_dir, seed)

        metrics_csv_path = self.output_path / "metrics_results.csv"
        self.mlflow_logger.log_test_metrics(fold, y_pred_test, partition, seed, _append_csv_row, metrics_csv_path)
        self.mlflow_logger.log_duration(history, _append_csv_row, metrics_csv_path, partition, seed)

    def _train_dl_replicate(self, fold, model, run_dir: Path, seed: int, g: torch.Generator):
        """Devuelve (y_pred_train, y_pred_val, y_pred_test, history) --
        _finalize_replicate hace el resto (guardado + logueo), común con
        _train_ml_replicate."""
        model.to(self.device)

        optimizer = self._setup_optimizer(model)
        criterion = self._setup_criterion()

        trainer_cfg = self.config.get("trainer", {}).get("params", {})
        preload_to_device = trainer_cfg.get("preload_to_device", False)

        train_loader, val_loader = self._setup_dl_dataloaders(fold, g, preload_to_device)

        weights_dir = run_dir / "weights"
        callbacks = self._setup_dl_callbacks(weights_dir, seed, optimizer, has_val=val_loader is not None)

        trainer = Trainer_DL(
            model=model, loss_fn=criterion, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            max_epochs=trainer_cfg.get("max_epochs", 5),
            callbacks=callbacks,
            preload_to_device=preload_to_device,
            device=self.device,
        )
        trainer.train()

        # --- Inferencia explícita post-training, órdenes preservados ---
        train_eval_loader = self._setup_single_loader("train_eval", fold.X_train, fold.y_train, g=None, preload_to_device=preload_to_device)
        y_pred_train = trainer.infer(dataloader=train_eval_loader, probability=True).cpu().numpy()

        y_pred_val = None
        if fold.X_val is not None:
            val_eval_loader = self._setup_single_loader("val", fold.X_val, fold.y_val, g=None, preload_to_device=preload_to_device)
            y_pred_val = trainer.infer(dataloader=val_eval_loader, probability=True).cpu().numpy()

        test_loader = self._setup_single_loader("test", fold.X_test, fold.y_test, g=None, preload_to_device=preload_to_device)
        y_pred_test = trainer.infer(dataloader=test_loader, probability=True).cpu().numpy()

        return y_pred_train, y_pred_val, y_pred_test, trainer.get_history()

    def _train_ml_replicate(self, fold, model, run_dir: Path, seed: int):
        """Análogo a _train_dl_replicate pero para Trainer_ML -- sin
        DataLoaders, sin device, sin history (get_history() de Trainer_ML
        siempre devuelve {}). TODO: guardado de pesos (joblib, no .pth) y
        checkpoint equivalente a CustomCheckpoint para modelos
        sklearn-like -- pendiente de definir."""
        trainer = Trainer_ML(model, fold.X_train, fold.y_train, X_val=fold.X_val, y_val=fold.y_val)
        trainer.train()

        y_pred_train = trainer.infer(fold.X_train, probability=True)
        y_pred_val = trainer.infer(fold.X_val, probability=True) if fold.X_val is not None else None
        y_pred_test = trainer.infer(fold.X_test, probability=True)

        return y_pred_train, y_pred_val, y_pred_test, trainer.get_history()