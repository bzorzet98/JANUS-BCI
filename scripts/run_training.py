"""
Punto de entrada para una corrida de training (script_type == "train_dl"
o "train_ml"). Instancia TrainingOrchestrator (clase, ver Paso 4 de la
sesión de diseño) -- este archivo es un entrypoint delgado, toda la
lógica de dominio (Schema, DataProvider, guardado, MLflow) vive en
src/training/orchestrator.py.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ProgressReporter, RichProgressReporter
from src.training.orchestrator_training import TrainingOrchestrator


def run_training(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
    device: str = "cpu",
) -> None:
    orchestrator = TrainingOrchestrator(
        config, docker_image=docker_image, reporter=reporter,
        force=force, overwrite_existing=overwrite_existing,
        device=device
    )
    orchestrator.run()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Corre un barrido de training (DL o ML).")
    parser.add_argument("--config", required=True, 
                        help="Path al JSON de config de training.")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu. Si se omite, autodetecta.")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--progress", choices=["none", "rich"], default="rich",
                         help="Progreso en terminal: 'rich' para tabla+barra en vivo, 'none' headless.")
    parser.add_argument("--force", action="store_true",
                         help="Ignora script_progress.csv y reprocesa todo, incluso lo ya marcado success.")
    parser.add_argument("--overwrite-existing", action="store_true",
                         help="Si el config guardado difiere del actual, sobrescribe el trío de "
                              "reproducibilidad y reinicia script_progress.csv/metrics_results.csv "
                              "en vez de rechazar.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    script_type = config.get("script_type")
    if script_type not in ("train_dl", "train_ml"):
        raise ValueError(
            f"run_training.py espera script_type 'train_dl' o 'train_ml', el config tiene "
            f"'{script_type}'. Usar el script correcto para ese script_type "
            f"(run_preprocessing.py, run_testing.py, etc.)."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reporter = RichProgressReporter() if args.progress == "rich" else ProgressReporter()
    run_training(config, args.docker_image, reporter, force=args.force, 
                 overwrite_existing=args.overwrite_existing, device=device)


if __name__ == "__main__":
    main()