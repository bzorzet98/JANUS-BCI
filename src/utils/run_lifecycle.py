"""
Componente compartido de reanudación/sobrescritura (PROTOCOL.md sección 7).

Generalizado desde la lógica que vivía inline en
`src/preprocessing/database_preprocessor.py` (`_check_config_compatibility`,
`_load_done_partitions`, `_filter_pending`, `_reset_output_for_overwrite`).
Usado tanto por `EEGDatabasePreprocessor` como por los orquestadores de
training/testing (`src/training/orchestrator.py`,
`src/training/orchestrator_testing.py`) — PROTOCOL.md exige "un solo
componente compartido... no reimplementado por cada orquestador".

Por qué `still_present` se inyecta en vez de estar hardcodeado acá: cada
dominio guarda "¿el output de esta partición sigue en disco?" de forma
distinta. Preprocessing lo resuelve con una columna `output_path` propia en
su CSV (formato libre). `script_progress.csv` (PROTOCOL.md sección 6) tiene
un schema FIJO de 4 columnas sin lugar para eso — ahí la pregunta se
responde mirando si existe el árbol de carpetas de la partición. Este
componente no puede asumir ninguna de las dos formas, así que el caller
decide.
"""
import csv
import datetime
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.datetime.now().isoformat()


def append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append + flush por fila (abre/escribe/cierra) — nada se acumula en
    memoria, así un barrido que se cae a mitad de camino deja consultable
    todo lo ya procesado. Promovido desde
    `src/preprocessing/database_preprocessor.py` (antes módulo-level ahí)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def drop_csv_rows(path: Path, column: str, value: str) -> None:
    """Borra de un CSV de formato largo las filas con `row[column] == value`.
    Pensado para reprocesar una partición (reintento tras un 'failed', o
    --force) sin duplicar sus filas: antes de volver a appendear, se
    descartan las que dejó la pasada anterior de ESA partición. Se lee y
    reescribe como texto (csv, no pandas) para no alterar el formato de
    las filas que quedan; la escritura va a un temporal + replace, así un
    corte a mitad de camino nunca deja el CSV truncado."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return
        rows = [row for row in reader if row.get(column) != value]
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def clear_sweep_outputs(output_path: Path, csv_names: Tuple[str, ...] = ("metrics_results.csv",)) -> None:
    """`extra_cleanup` de --overwrite-existing para training/testing: borra
    los CSV de métricas y TODAS las subcarpetas de `output_path` (una por
    partición: pesos, outputs, marcadores). Sin esto, archivos de la
    identidad anterior -- ej. pesos de seeds que el config nuevo ya no usa
    -- quedarían mezclados con los nuevos en la misma carpeta de partición
    (PROTOCOL.md sección 7: nunca se mezclan identidades distintas). El
    trío (config.json/.git_commit/.docker_image) no se toca acá: el caller
    lo reescribe después."""
    for name in csv_names:
        csv_path = output_path / name
        if csv_path.exists():
            csv_path.unlink()
    for child in output_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def diff_values(old: Any, new: Any, path: str = "identity") -> List[str]:
    """Diff recursivo genérico, para armar el mensaje de 'qué cambió' cuando
    la identidad guardada no coincide con la nueva. Promovido desde
    `database_preprocessor.py` (antes módulo-level ahí, ahora reusado como
    default de `check_identity_or_raise`)."""
    diffs: List[str] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                diffs.append(f"{path}.{key}: agregado ({new[key]!r})")
            elif key not in new:
                diffs.append(f"{path}.{key}: eliminado (era {old[key]!r})")
            elif old[key] != new[key]:
                diffs.extend(diff_values(old[key], new[key], f"{path}.{key}"))
    elif old != new:
        diffs.append(f"{path}: {old!r} -> {new!r}")
    return diffs


class RunLifecycleManager:
    """Reanudación/sobrescritura para un barrido cuyo output vive bajo
    `output_path`. No sabe nada del dominio (BCI, imágenes, etc.) ni del
    formato de config — todo lo específico se inyecta por el caller.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        progress_filename: str = "script_progress.csv",
        identity_filename: str = "config.json",
        force: bool = False,
        overwrite_existing: bool = False,
        emit_running_row: bool = False,
    ):
        self.output_path = output_path
        self.progress_path = output_path / progress_filename
        self.identity_path = output_path / identity_filename
        self.force = force
        self.overwrite_existing = overwrite_existing
        # Default False: adoptar este componente en EEGDatabasePreprocessor
        # no cambia su CSV actual, que nunca emitió una fila 'running'
        # intermedia. PROTOCOL.md sección 6 sí contempla 'running' en el
        # vocabulario de status -- training/testing lo activan.
        self.emit_running_row = emit_running_row

    # --- identidad / overwrite -------------------------------------------------

    def check_identity_or_raise(
        self,
        new_config: dict,
        *,
        extract_identity: Callable[[dict], Any] = lambda c: c,
        diff_fn: Optional[Callable[[Any, Any], List[str]]] = None,
        extra_cleanup: Optional[Callable[[], None]] = None,
        context_label: str = "la corrida",
    ) -> None:
        """Si `identity_path` no existe todavía, no hay nada que comparar
        (primera corrida). Si existe y la identidad extraída coincide, sigue
        de largo. Si difiere: `--overwrite-existing` sobrescribe sin
        preguntar; con TTY interactiva pregunta confirmación; sin TTY
        (batch/cron) rechaza duro -- nunca bloquea esperando input
        (PROTOCOL.md sección 7)."""
        if diff_fn is None:
            diff_fn = diff_values
        if not self.identity_path.exists():
            return

        existing_config = json.loads(self.identity_path.read_text())
        existing_identity = extract_identity(existing_config)
        new_identity = extract_identity(new_config)
        if existing_identity == new_identity:
            return

        diffs = diff_fn(existing_identity, new_identity)
        message = (
            f"{context_label} existente en {self.output_path} tiene una "
            f"identidad distinta a la de este config:\n"
            + "\n".join(f"  - {d}" for d in diffs)
            + "\nUsá una identidad distinta (recipe_name/label/etc.) para una "
              "corrida nueva, o --overwrite-existing si estás seguro de "
              "reemplazar la corrida anterior."
        )

        if self.overwrite_existing:
            logger.warning(
                "Identidad distinta detectada — sobrescribiendo por --overwrite-existing."
            )
            self.reset_progress(extra_cleanup=extra_cleanup)
            return

        if sys.stdin.isatty():
            answer = input(f"{message}\n¿Sobrescribir de todas formas? [y/N]: ").strip().lower()
            if answer == "y":
                self.reset_progress(extra_cleanup=extra_cleanup)
                return
            raise RuntimeError(f"Corrida cancelada por el usuario.\n{message}")

        # Sin TTY (batch/cron): nunca bloquear esperando input, rechazar duro.
        raise RuntimeError(message)

    def reset_progress(self, extra_cleanup: Optional[Callable[[], None]] = None) -> None:
        """Borra `progress_path`. `extra_cleanup`, si se da, corre después --
        para callers cuyos artefactos de salida viven en más lugares que el
        CSV de progreso (ej. preprocessing borra también stage_metrics.csv y
        las detail tables; training tendría que borrar recursivamente cada
        carpeta de replicate)."""
        if self.progress_path.exists():
            self.progress_path.unlink()
        if extra_cleanup is not None:
            extra_cleanup()

    # --- resume ------------------------------------------------------------

    def load_success_partitions(self, still_present: Callable[[str], bool]) -> set:
        """Partitions con status == 'success' en `progress_path` que además
        tienen su output todavía presente en disco según `still_present`. Si
        una partition aparece más de una vez (barridos --force previos), se
        toma la fila más reciente."""
        if not self.progress_path.exists():
            return set()
        progress = pd.read_csv(self.progress_path)
        if progress.empty:
            return set()
        latest = progress.drop_duplicates(subset="partition", keep="last")
        done = set()
        for _, row in latest.iterrows():
            if row.get("status") != "success":
                continue
            partition = row["partition"]
            if still_present(partition):
                done.add(partition)
        return done

    def filter_pending(self, all_partitions: list, done: set, key: Callable[[Any], str] = lambda x: x) -> list:
        """Descarta de `all_partitions` lo que ya está en `done`. `key`
        extrae el identificador de partition de cada elemento -- por default
        el elemento es directamente ese identificador (caso training/
        testing); preprocessing pasa `key=lambda item: item[2]` porque sus
        elementos son tuplas `(subject, session, partition)`."""
        return [item for item in all_partitions if key(item) not in done]

    def prepare_partitions(
        self,
        all_partitions: list,
        still_present: Callable[[str], bool],
        key: Callable[[Any], str] = lambda x: x,
    ) -> list:
        """Envoltorio de conveniencia: `force` salta el resume por completo;
        si no, `load_success_partitions` + `filter_pending`."""
        if self.force:
            return list(all_partitions)
        done = self.load_success_partitions(still_present)
        return self.filter_pending(all_partitions, done, key=key)

    # --- escritura de progreso ----------------------------------------------

    def mark_start(self, partition: str) -> str:
        """Devuelve `ts_start` (ISO 8601). Si `emit_running_row`, además
        appendea de inmediato una fila `{partition, status:'running',
        timestamp_start, timestamp_end:''}` -- hace observable
        `script_progress.csv` a mitad de un barrido largo, en vez de que
        solo aparezca la fila final al terminar esa partition."""
        ts_start = _now()
        if self.emit_running_row:
            append_csv_row(self.progress_path, {
                "partition": partition,
                "status": "running",
                "timestamp_start": ts_start,
                "timestamp_end": "",
            })
        return ts_start

    def mark_result(self, partition: str, status: str, ts_start: str) -> None:
        """Appendea la fila terminal (contrato PROTOCOL.md sección 6:
        `partition`/`status`/`timestamp_start`/`timestamp_end`)."""
        append_csv_row(self.progress_path, {
            "partition": partition,
            "status": status,
            "timestamp_start": ts_start,
            "timestamp_end": _now(),
        })
