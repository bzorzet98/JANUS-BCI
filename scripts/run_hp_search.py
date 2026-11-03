"""
Punto de entrada para una búsqueda de hiperparámetros. Instancia
HPSearchOrchestrator (ver src/training/orchestrator_hp_search.py) --
entrypoint delgado, toda la lógica vive ahí.

Recibe DOS configs, a diferencia de los demás entrypoints:
  --hp_search_config: la grilla (grid, dev_units, fixed_overrides)
  --base_config: el config de training real sobre el que se arma cada
                 combinación (mismo formato que usa run_training.py)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.progress_reporter import ProgressReporter, RichProgressReporter
from src.training.orchestrator_hp_search import HPSearchOrchestrator


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Corre una búsqueda de hiperparámetros.")
    parser.add_argument("--hp_search_config", required=True, help="Path al JSON de configuración de búsqueda (grid, dev_units, fixed_overrides).")
    parser.add_argument("--base_config", required=True, help="Path al JSON de config de training base sobre el que se arma cada combinación.")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu. Si se omite, autodetecta.")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--progress", choices=["none", "rich"], default="rich",
                         help="Progreso en terminal: 'rich' para tabla+barra en vivo, 'none' headless.")
    args = parser.parse_args()

    hp_search_config = json.loads(Path(args.hp_search_config).read_text())
    base_config = json.loads(Path(args.base_config).read_text())

    reporter = RichProgressReporter() if args.progress == "rich" else ProgressReporter()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    orchestrator = HPSearchOrchestrator(
        hp_search_config, base_config, docker_image=args.docker_image, reporter=reporter, device=device
    )
    orchestrator.run()


if __name__ == "__main__":
    main()