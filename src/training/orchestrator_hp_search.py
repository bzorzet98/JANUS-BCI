"""
`HPSearchOrchestrator` -- búsqueda de hiperparámetros, reemplaza el
sistema viejo (generate_hp_grid.py + run_hp_search.py + dotted-path
--overrides) adaptado al diseño nuevo.

Diferencias de fondo respecto al sistema viejo:
- Cada eje de la grilla es un BLOQUE DE CONFIG COMPLETO (ej. todo
  "optimizer", todo "torch_dataloader"), no un dotted-path anidado --
  elimina la fragilidad de indexar por posición de lista (el viejo
  "callbacks.1.params.patience" asumía que EarlyStopping siempre estaba
  en la posición 1).
- Instancia un TrainingOrchestrator COMPLETO por combinación -- reusa
  TODO el lifecycle/guardado/MLflow que ya existe, no reimplementa nada.
- Los resultados de búsqueda quedan en un árbol de resultados
  COMPLETAMENTE SEPARADO del de producción: strategy_name gana el sufijo
  "_HPSearch", así "métricas exploratorias de búsqueda" nunca se mezclan
  en el mismo metrics_results.csv que "resultados finales reportables"
  (metrics_results.csv vive DENTRO de cada label, ver fix de path
  aplicado a TrainingOrchestrator).
- dev_units se inyecta directo en schema.params.units_to_test -- ya no
  hace falta "general_script_config.subjects_to_test" como override
  genérico, ese campo ya no existe.
- hp_combinations.csv (formato largo: combo_hash, axis_name,
  axis_value_json) queda en el nivel {project}/{strategy}_HPSearch/
  {recipe}/{dataset_dir}/ -- un nivel arriba de cada label individual,
  para trackeo rápido en disco. combo_hash también se taggea en MLflow
  (vía project_config.label, que ya se usa como tag) -- MLflow sigue
  siendo la fuente real de análisis exploratorio (curvas, comparación
  entre combinaciones); este CSV es solo para no tener que abrir MLflow
  para saber qué hiperparámetros corresponden a qué carpeta en disco.
"""
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.training.orchestrator_training import TrainingOrchestrator
from src.utils.paths import PATHS

logger = logging.getLogger(__name__)


def combo_hash(axis_dict: dict) -> str:
    """Hash corto y estable (no depende del orden de las keys) -- mismo
    criterio que combo_id_from_overrides del sistema viejo
    (generate_hp_grid.py)."""
    canonical = json.dumps(axis_dict, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:10]


def _set_nested(config: dict, dotted_path: str, value: Any) -> None:
    """Solo para fixed_overrides -- unos pocos campos conocidos y NO
    indexados por posición de lista (ej. "orchestrator.model_init_seed"),
    a diferencia de la grilla en sí, que siempre reemplaza bloques
    completos. Si algún fixed_override necesitara indexar una lista por
    posición, es señal de que ese campo debería declararse como bloque
    completo en 'grid' en vez de en 'fixed_overrides' -- no se agrega
    soporte para indexar listas acá a propósito."""
    parts = dotted_path.split(".")
    node = config
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


class HPSearchOrchestrator:
    """Coordina una búsqueda de hiperparámetros completa: genera un
    config por combinación de la grilla, instancia y corre un
    TrainingOrchestrator por cada uno."""

    def __init__(
        self, hp_search_config: dict, base_config: dict,
        docker_image: Optional[str] = None, reporter=None, device: str = "cpu"
    ):
        self.hp_search_config = hp_search_config
        self.base_config = base_config
        self.docker_image = docker_image
        self.reporter = reporter
        self.device = device

        base_project_config = base_config["project_config"]
        self.project_name = base_project_config["project_name"]
        self.strategy_name = f"{base_project_config['strategy_name']}_HPSearch"
        self.recipe_name = base_project_config["recipe_name"]
        self.dataset_dir_name = f"{base_project_config['database_name']}_{base_project_config['database_session']}"

        self.output_path = (
            PATHS.results_root / self.project_name / self.strategy_name /
            self.recipe_name / self.dataset_dir_name
        )

    def _build_combo_configs(self) -> Tuple[List[dict], List[Dict[str, Any]]]:
        """Producto cartesiano de grid. Cada combinación es una COPIA
        completa de base_config con los bloques de la grilla
        reemplazados enteros (nunca dotted-path indexado por posición).
        Devuelve (lista de configs, filas para hp_combinations.csv)."""
        import itertools

        base_project_config = self.base_config["project_config"]
        base_label = base_project_config["label"]
        dev_units = self.hp_search_config["dev_units"]
        fixed_overrides = self.hp_search_config.get("fixed_overrides", {})
        grid = self.hp_search_config["grid"]

        axis_names = list(grid.keys())
        value_lists = [grid[name] for name in axis_names]

        combo_configs = []
        combo_rows = []

        for combo_values in itertools.product(*value_lists):
            combo_config = copy.deepcopy(self.base_config)

            combo_config["project_config"]["strategy_name"] = self.strategy_name
            combo_config["schema"]["params"]["units_to_test"] = dev_units

            axis_dict = dict(zip(axis_names, combo_values))
            hash_ = combo_hash(axis_dict)
            combo_config["project_config"]["label"] = f"{base_label}_hp{hash_}"

            for axis_name, value in axis_dict.items():
                combo_config[axis_name] = value  # reemplazo de BLOQUE completo
                combo_rows.append({
                    "combo_hash": hash_, "axis_name": axis_name, "axis_value_json": json.dumps(value),
                })

            for path, value in fixed_overrides.items():
                _set_nested(combo_config, path, value)

            combo_configs.append(combo_config)

        return combo_configs, combo_rows

    def _write_hp_combinations_csv(self, combo_rows: List[Dict[str, Any]]) -> None:
        import csv as csv_module

        csv_path = self.output_path / "hp_combinations.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=["combo_hash", "axis_name", "axis_value_json"])
            if write_header:
                writer.writeheader()
            for row in combo_rows:
                writer.writerow(row)

    def run(self) -> None:
        combo_configs, combo_rows = self._build_combo_configs()

        self.output_path.mkdir(parents=True, exist_ok=True)
        self._write_hp_combinations_csv(combo_rows)

        logger.info(
            "HPSearch: %d combinaciones generadas, resultados en %s",
            len(combo_configs), self.output_path,
        )

        for i, combo_config in enumerate(combo_configs):
            label = combo_config["project_config"]["label"]
            logger.info("--- Combinación %d/%d (label=%s) ---", i + 1, len(combo_configs), label)

            orchestrator = TrainingOrchestrator(
                combo_config, docker_image=self.docker_image, reporter=self.reporter,
                device = self.device
            )
            orchestrator.run()

        logger.info("--- HPSearch Finished: %d combinaciones ---", len(combo_configs))