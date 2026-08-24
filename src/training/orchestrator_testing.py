"""
`TestingOrchestrator` -- reemplaza run_testing_sweep (función vieja).
Evalúa modelos YA entrenados -- posiblemente con un dataset/dataloader de
ablación distinto al usado en training -- contra la estrategia/dataset
propios de ESTE config de test, que pueden ser distintos de los del
training de origen (declarados en config["model_source"]).

Cambios de fondo respecto a run_testing_sweep:
- Clase (no función), mismo motivo que TrainingOrchestrator (Capa 5,
  "orquestador debe ser clase, no función" -- legibilidad).
- project_config["recipe_name"] en vez de "model_to_test" -- mismo rename
  aplicado a TrainingOrchestrator.
- output_path incluye project_name/label (mismo fix de path que
  TrainingOrchestrator -- antes strategy/recipe/database mezclaba
  resultados de distinto label/proyecto en el mismo árbol).
- _compute_test_metrics hardcodeada -> MetricsStrategy resuelta vía
  import_class (o ClassificationMetrics() default) -- generaliza a N
  clases, arregla el AUC/recall/precision atados a 2 clases fijas.
- MATCHER_REGISTRY["WithinSubjectMatcher"] -> "WithinUnitMatcher".
- config["loaded_model_config"] -> config["model_source"] (rename, mismo
  contenido salvo que ya NO declara schema_name a mano -- se lee del
  config.json real de la carpeta de origen, nunca escrito por el usuario:
  ver _read_schema_name_from_origin).
- model_source.script_type ("train_dl"/"train_ml") habilita dispatch real
  Tester_DL/Tester_ML -- antes SIEMPRE instanciaba Tester_DL.
- DataProvider en vez de dataset.flatten_subject_data directo -- mismo
  BCIDataProviderAdapter que usa TrainingOrchestrator, agnóstico de cómo
  se llama "unidad" en el dominio.
- Nombrado de partición vía "unit_{id:02d}" explícito (no depende de
  ningún Schema real instanciado, testing no arma Schema -- pero preserva
  el mismo formato de nombre que ya usan los Schemas within-unit, para
  que WithinUnitMatcher pueda interpretarlo sin ambigüedad).
"""
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch

from src.training.data_provider import DataProvider
from src.training.core.device import obtain_cuda_device
from src.training.core.tester import Tester_DL, Tester_ML
from src.training.metrics.classification import ClassificationMetrics
from src.training.utils import create_dataloader, create_dataset, encode_labels
from src.training.weights.matcher import MATCHER_REGISTRY
from src.training.weights.resolver import TrainedModelResolver
from src.utils.imports import import_class
from src.utils.paths import PATHS
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, _append_csv_row

logger = logging.getLogger(__name__)

TESTER_REGISTRY = {
    "train_dl": Tester_DL,
    "train_ml": Tester_ML,
}


def _testing_identity(config: dict) -> dict:
    identity = copy.deepcopy(config)
    orchestrator_cfg = identity.get("orchestrator", {})
    orchestrator_cfg.pop("device", None)
    return identity


def _read_schema_name_from_origin(origin_output_path: Path) -> Optional[str]:
    """Lee el config.json real del training de origen (trío de
    reproducibilidad) y extrae schema.class_name -- NUNCA se declara a
    mano en model_source, para que no pueda contradecir lo que
    REALMENTE se usó para entrenar. Devuelve None si el config de origen
    no tiene bloque 'schema' (ej. entrenado con la API vieja, antes de
    esta refactorización) -- se registra igual, sin bloquear el testing."""
    config_path = origin_output_path / "config.json"
    if not config_path.exists():
        return None
    origin_config = json.loads(config_path.read_text())
    return origin_config.get("schema", {}).get("class_name")


class TestingOrchestrator:
    """Coordina un barrido de testing/evaluación sobre modelos ya
    entrenados. Instanciar una vez por corrida (un config), llamar .run()."""

    def __init__(
        self, config: dict, docker_image: Optional[str] = None,
        reporter=None, force: bool = False, overwrite_existing: bool = False,
        device: str = "cpu",
    ):
        self.config = config
        self.docker_image = docker_image
        self.reporter = reporter
        self.force = force
        self.overwrite_existing = overwrite_existing
        self.device = device
        
        project_config = config["project_config"]
        self.project_name = project_config["project_name"]
        self.strategy_name = project_config["strategy_name"]
        self.recipe_name = project_config["recipe_name"]
        self.database_name = project_config["database_name"]
        self.database_session = project_config["database_session"]
        self.label = project_config["label"]

        self.model_source = config["model_source"]
        self.model_script_type = self.model_source.get("script_type", "train_dl")

        dataset_dir_name = f"{self.database_name}_{self.database_session}"
        self.output_path = (
            PATHS.results_root / self.project_name / self.strategy_name /
            self.recipe_name / dataset_dir_name / self.label
        )

        self.provider = None
        self.lifecycle: Optional[RunLifecycleManager] = None
        self.metrics_strategy = None

    def _setup_provider(self) -> DataProvider:
        """Instancia el dataset real vía import_class (config['databases']),
        lo envuelve en el adapter configurado en config['data_provider'].
        NO hay default BCI-específico: 'data_provider' es un campo
        OBLIGATORIO en el config -- este orquestador (candidato a JANUS/) no
        debe asumir ningún adapter concreto de ningún dominio."""
        dataset_cfg = self.config["databases"]
        dataset = import_class(dataset_cfg["class_name"], dataset_cfg["module_name"])(**dataset_cfg["params"])

        provider_cfg = self.config.get("data_provider")
        if provider_cfg is None:
            raise ValueError(
                "config['data_provider'] es obligatorio -- no hay default agnóstico de dominio. "
            )

        provider_class = import_class(provider_cfg["class_name"], provider_cfg["module_name"])
        params = dict(provider_cfg.get("params", {}))
        params["dataset"] = dataset
        return provider_class(**params)

    def _setup_metrics_strategy(self):
        metrics_cfg = self.config.get("metrics_strategy")
        if metrics_cfg is None:
            return ClassificationMetrics()
        metrics_class = import_class(metrics_cfg["class_name"], metrics_cfg["module_name"])
        return metrics_class(**metrics_cfg.get("params", {}))

    def _origin_output_path(self) -> Path:
        """Path a la carpeta de origen (training) que generó los pesos a
        testear -- construido con la MISMA convención de path que
        TrainingOrchestrator, a partir de model_source."""
        origin_dataset_dir = f"{self.model_source['database_name']}_{self.model_source['database_session']}"
        return (
            PATHS.results_root / self.project_name / self.model_source["strategy_name"] /
            self.model_source["recipe_name"] / origin_dataset_dir / self.model_source["label"]
        )

    def run(self) -> None:
        self.provider = self._setup_provider()
        self.metrics_strategy = self._setup_metrics_strategy()

        config_snapshot = copy.deepcopy(self.config)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.lifecycle = RunLifecycleManager(
            self.output_path,
            force=self.force, overwrite_existing=self.overwrite_existing,
            emit_running_row=False,
        )
        self.lifecycle.check_identity_or_raise(
            new_config=config_snapshot,
            extract_identity=_testing_identity,
            context_label=f"la corrida de testing en {self.output_path}",
        )
        log_reproducibility_trio(self.output_path, config_snapshot, self.docker_image)

        origin_output_path = self._origin_output_path()
        schema_name = _read_schema_name_from_origin(origin_output_path)
        if schema_name is None:
            logger.warning(
                "No se pudo leer schema.class_name del config.json de origen (%s) -- "
                "el training de origen puede haber usado la API vieja. Se continúa igual, "
                "sin ese dato de trazabilidad.", origin_output_path,
            )

        resolver = TrainedModelResolver(PATHS.results_root, self.model_source)
        all_weights = list(resolver.resolve_weights())
        matcher = MATCHER_REGISTRY["WithinUnitMatcher"]()

        # Testing siempre evalúa todas las unidades del PROVIDER de test
        # (no hay "units_to_test" acá -- el filtro real es "¿tiene esta
        # unidad pesos entrenados emparejados?", resuelto vía resolver+matcher).
        all_units = self.provider.list_units()
        partition_to_unit = {f"unit_{u:02d}": u for u in all_units}

        matches_by_partition: Dict[str, list] = {}
        for partition in partition_to_unit:
            matched = matcher.match(all_weights, partition)
            if matched:
                matches_by_partition[partition] = matched

        if not matches_by_partition:
            logger.info(
                "Ninguna unidad de %s tiene pesos entrenados emparejados bajo %s -- nada para hacer.",
                self.database_name, self.model_source,
            )
            return

        def _still_present(partition: str) -> bool:
            partition_dir = self.output_path / partition
            if not partition_dir.exists():
                return False
            return all(
                (partition_dir / f"seed_{weights.metadata["seed"]}_outputs.csv").exists()
                for weights in matches_by_partition[partition]
            )

        pending = self.lifecycle.prepare_partitions(list(matches_by_partition.keys()), still_present=_still_present)

        if not pending:
            logger.info("Todo ya procesado (%d particiones) -- nada para hacer.", len(matches_by_partition))
            return

        mlflow.set_experiment(f"{self.project_name}/{self.strategy_name}/{self.recipe_name}/{self.database_name}_{self.database_session}/{self.label}")

        if self.reporter:
            self.reporter.on_start(len(pending), meta={
                "strategy_name": self.strategy_name, "recipe_name": self.recipe_name,
                "database_name": self.database_name, "n_partitions": len(pending),
            })

        for partition in pending:
            self._run_partition(partition, partition_to_unit[partition], matches_by_partition[partition], self.device, schema_name)

        logger.info("--- Testing Finished ---")
        if self.reporter:
            self.reporter.on_finish({"total": len(pending)})

    def _run_partition(self, partition: str, unit_id, matched_weights: list, device, schema_name: Optional[str]) -> None:
        ts_start = self.lifecycle.mark_start(partition)
        if self.reporter:
            self.reporter.on_run_start(partition)
        partition_status = "success"

        try:
            X, y = self.provider.get_unit(unit_id)
            y_encoded, _ = encode_labels(y)

            X_t = torch.tensor(X, dtype=torch.float).to(device)
            y_t = torch.tensor(y_encoded, dtype=torch.long).to(device)

            test_dataset = create_dataset("test", X_t, y_t, self.config["torch_dataset"])
            test_loader = create_dataloader("test", test_dataset, self.config["torch_dataloader"], g=None)
            y_true = y_t.cpu().numpy()

            for weights in matched_weights:
                self._run_replicate(partition, weights, X_t, test_loader, y_true, device, schema_name)

            if self.reporter:
                self.reporter.on_run_result(partition, "success")

        except Exception as e:
            logger.error("Fallo cargando/evaluando %s: %s", partition, e)
            partition_status = "failed"

        self.lifecycle.mark_result(partition, partition_status, ts_start)

    def _run_replicate(self, partition: str, weights, X_t, test_loader, y_true, device, schema_name: Optional[str]) -> None:
        origin_seed = weights.metadata["segments"][-1]
        run_dir = self.output_path / partition

        model_cfg = self.config["model"]
        model = import_class(model_cfg["class_name"], model_cfg["module_name"])(**model_cfg["params"])

        try:
            criterion = None
            criterion_cfg = self.config.get("criterion")
            if criterion_cfg:
                criterion = import_class(criterion_cfg["class_name"], criterion_cfg["module_name"])(**criterion_cfg["params"])

            tags = {
                "project_name": self.project_name,
                "strategy_name": self.strategy_name,
                "database_name": self.database_name,
                "label": self.label,
                "partition": partition,
                "origin_partition": weights.partition,
                "origin_schema": schema_name or "unknown",
            }

            with mlflow.start_run(run_name=f"{partition}_{origin_seed}", tags=tags) as run:
                tester_class = TESTER_REGISTRY[self.model_script_type]

                if tester_class is Tester_DL:
                    model.load_state_dict(torch.load(weights.path, map_location=device))
                    model.to(device)
                    tester = Tester_DL(model, loss_fn=criterion)
                    y_pred_proba, loss = tester.test(test_loader, probability=True)
                    y_pred_proba = y_pred_proba.cpu().numpy()
                else:
                    import joblib
                    model = joblib.load(weights.path)
                    tester = Tester_ML(model)
                    y_pred_proba, loss = tester.test(X_t.cpu().numpy(), probability=True)

                y_pred = np.argmax(y_pred_proba, axis=1)

                log_reproducibility_trio(run_dir, self.config, self.docker_image)

                metrics_rows = self.metrics_strategy.compute(y_true, y_pred, y_pred_proba=y_pred_proba, split="test")
                for row in metrics_rows:
                    row["partition"] = partition
                    row["origin_seed"] = origin_seed
                    _append_csv_row(self.output_path / "metrics_results.csv", row)
                    mlflow.log_metric(row["metric_name"], row["value"])
                if loss is not None:
                    mlflow.log_metric("loss", float(loss))

                (run_dir / f".mlflow_run_id_{origin_seed}").write_text(run.info.run_id)

            if self.reporter:
                self.reporter.on_run_result(f"{partition}_{origin_seed}", "success")

        except Exception as e:
            logger.error("Fallo testeando %s/%s: %s", partition, origin_seed, e)
            if self.reporter:
                self.reporter.on_run_result(f"{partition}_{origin_seed}", "failed")