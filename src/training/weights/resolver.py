"""
`TrainedModelResolver` -- agnóstico a semántica de partición (principio 4
del refactor): recorre el árbol de resultados de un training ya corrido y
devuelve TODOS los pesos encontrados, sin asumir qué representa cada nivel
de carpeta más allá de "el primer segmento es la unidad que
`WithinSubjectMatcher` usa para matchear" (eso vive en el matcher, no acá).

Decisión de diseño (glob + filtro por script_progress.csv, no uno u otro
puro): un glob puro no sabe qué particiones fallaron; leer directamente
`script_progress.csv` reintroduce la semántica de path que el principio 4
pide evitar. Acá se hace glob de archivos de pesos y se cruza contra el
`script_progress.csv` del training de origen para descartar particiones
`failed`/`running`, sin asumir nada sobre la profundidad del árbol más allá
de "el primer segmento del path relativo es la partition que aparece en
script_progress.csv" (eso coincide con cómo la escribe
src/training/orchestrator.py).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import re
import logging
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class TrainedWeights:
    path: Path
    partition: str  # path relativo (sin el nombre de archivo) tal cual aparece en disco
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainedModelResolver:
    def __init__(
        self, results_root: Path, model_source: dict,
        weights_glob: str = "*_best.pth",
        seed_pattern: str = r"seed_(\d+)_best",
    ):
        """weights_glob: patrón de archivo a buscar. seed_pattern: regex
        con UN grupo de captura que extrae la seed del nombre de archivo
        -- ambos configurables porque el nombre depende de qué callback
        de guardado se use (hoy CustomCheckpoint con prefix="seed_{N}_",
        pero un callback nuevo podría nombrar distinto)."""
        dataset_dir_name = f"{model_source['database_name']}_{model_source['database_session']}"
        self.root = (
            Path(results_root) / model_source["project_name"] / model_source["strategy_name"] /
            model_source["recipe_name"] / dataset_dir_name / model_source["label"]
        )
        self.weights_glob = weights_glob
        self.seed_pattern = re.compile(seed_pattern)
        self._success_partitions = self._load_success_partitions()

    def _load_success_partitions(self) -> set:
        progress_path = self.root / "script_progress.csv"
        if not progress_path.exists():
            return set()
        df = pd.read_csv(progress_path)
        if df.empty:
            return set()
        latest = df.drop_duplicates(subset="partition", keep="last")
        return set(latest[latest["status"] == "success"]["partition"])

    def _extract_seed(self, path: Path) -> Optional[str]:
        match = self.seed_pattern.search(path.name)
        return match.group(1) if match else None

    def resolve_weights(self) -> Iterator[TrainedWeights]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob(self.weights_glob)):
            rel = path.relative_to(self.root)
            segments = rel.parts[:-1]
            if not segments:
                continue
            partition = segments[0]
            if partition not in self._success_partitions:
                continue

            seed = self._extract_seed(path)
            if seed is None:
                logger.warning(
                    "No se pudo extraer seed del archivo %s con el patrón %s -- se omite.",
                    path, self.seed_pattern.pattern,
                )
                continue

            yield TrainedWeights(
                path=path,
                partition=partition,
                metadata={"segments": segments, "seed": seed},
            )