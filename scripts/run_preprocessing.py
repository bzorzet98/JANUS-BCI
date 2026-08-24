"""
Punto de entrada para una corrida de preprocesamiento (script_type ==
"preprocessing"). Reemplaza la rama de preprocessing que antes vivía
dentro de run_production.py -- separado en su propio script porque los
flags/argumentos y la lógica de dominio no tienen por qué compartir
archivo con training/testing (ver sesión de diseño: "entrypoints
delgados separados, lifecycle compartido").

Delega directo a EEGDatabasePreprocessor
(src/preprocessing/database_preprocessor.py) -- este archivo no conoce
nada de la lógica de dominio del preprocesamiento en sí.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import EEGDatabasePreprocessor, ProgressReporter, RichProgressReporter
from src.utils.imports import import_class


def run_preprocessing(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """El trío de reproducibilidad y el manifest los escribe el propio
    EEGDatabasePreprocessor.run()."""
    db_config = config["database"]
    dataset_cls = import_class(db_config["class_name"], db_config["module_name"])
    dataset = dataset_cls(**db_config["kwargs"])

    orchestrator = EEGDatabasePreprocessor(
        dataset, config["preprocessing_name"], config, docker_image=docker_image, reporter=reporter,
        force=force, overwrite_existing=overwrite_existing,
    )
    orchestrator.run()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Corre un barrido de preprocesamiento EEG.")
    parser.add_argument("--config", required=True, help="Path al JSON de config de preprocesamiento.")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--progress", choices=["none", "rich"], default="rich",
                         help="Progreso en terminal: 'rich' para tabla+barra en vivo, 'none' headless.")
    parser.add_argument("--force", action="store_true",
                         help="Ignora run_registry.csv y reprocesa todo, incluso lo ya marcado success.")
    parser.add_argument("--overwrite-existing", action="store_true",
                         help="Si el preprocessing_pipeline guardado difiere del actual, sobrescribe "
                              "config.json/.git_commit/.docker_image y reinicia run_registry.csv, "
                              "stage_metrics.csv y las detail tables en vez de rechazar.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    script_type = config.get("script_type")
    if script_type != "preprocessing":
        raise ValueError(
            f"run_preprocessing.py espera script_type == 'preprocessing', el config tiene "
            f"'{script_type}'. Usar el script correcto para ese script_type "
            f"(run_training.py, run_testing.py, etc.)."
        )

    reporter = RichProgressReporter() if args.progress == "rich" else ProgressReporter()
    run_preprocessing(config, args.docker_image, reporter, force=args.force, overwrite_existing=args.overwrite_existing)


if __name__ == "__main__":
    main()