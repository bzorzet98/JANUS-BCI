"""
Punto de entrada para una corrida de testing/evaluación (script_type ==
"test_dl" o "test_ml"). Evalúa modelos ya entrenados -- posiblemente con
un dataset/ablación distinto al de training -- no reentrena nada.

NOTA IMPORTANTE: src/training/orchestrator_testing.py (run_testing_sweep)
TODAVÍA NO fue migrado al diseño nuevo (TrainingSchema/DataProvider/
MetricsStrategy) -- sigue con la API vieja completa (_compute_test_metrics
hardcodeada, WithinSubjectMatcher en vez de WithinUnitMatcher, sin Schema
ni DataProvider). Es deuda de refactorización pendiente, del mismo
tamaño que la que se le hizo a orchestrator.py -- no forma parte de la
sesión de diseño actual. Este script queda funcional para lo que
orchestrator_testing.py YA soporta hoy, pero heredará esa deuda hasta
que se haga esa migración.
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
from src.training.orchestrator_testing import run_testing_sweep


def run_testing(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
    device: str = "cpu",
) -> None:
    run_testing_sweep(config, docker_image, reporter, force=force, overwrite_existing=overwrite_existing)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Corre un barrido de testing/evaluación sobre modelos ya entrenados.")
    parser.add_argument("--config", required=True, help="Path al JSON de config de testing.")
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
    if script_type not in ("test_dl", "test_ml"):
        raise ValueError(
            f"run_testing.py espera script_type 'test_dl' o 'test_ml', el config tiene "
            f"'{script_type}'. Usar el script correcto para ese script_type "
            f"(run_preprocessing.py, run_training.py, etc.)."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reporter = RichProgressReporter() if args.progress == "rich" else ProgressReporter()
    run_testing(config, args.docker_image, reporter, force=args.force, 
                overwrite_existing=args.overwrite_existing, device=device)


if __name__ == "__main__":
    main()