"""
Progreso interactivo para EEGDatabasePreprocessor (DESIGN.md, cambio
acotado sobre el orquestador existente). `ProgressReporter` es el
contrato — no-op por default para no romper cron ni logs de servidor.
`RichProgressReporter` es la única implementación real, para debug
interactivo en terminal.
"""
from typing import Any, Dict, List, Optional


class ProgressReporter:
    """No-op por default. EEGDatabasePreprocessor siempre tiene un
    reporter (nunca None) — este es el que usa cuando no se pidió uno
    explícito."""

    def on_start(self, total: int, meta: Optional[dict] = None) -> None:
        pass

    def on_run_start(self, partition: str) -> None:
        pass

    def on_run_result(self, partition: str, status: str) -> None:
        pass

    def on_finish(self, summary: dict) -> None:
        pass


class RichProgressReporter(ProgressReporter):
    """Panel de metadata fijo (impreso una vez, fuera del Live) + barra de
    progreso global (una unidad de avance por partition resuelta, vía
    on_run_result). Los failures se imprimen fuera del Live apenas ocurren
    -- quedan arriba de la barra en vez de perderse en una fila que la
    barra termina tapando."""

    def __init__(self, console: Optional[Any] = None) -> None:
        from rich.console import Console

        self.console = console or Console()
        self._live = None
        self._progress = None
        self._task_id = None
        self._failed: List[str] = []
        self._start_time = 0.0

    def on_start(self, total: int, meta: Optional[dict] = None) -> None:
        import time

        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        meta = meta or {}
        self._start_time = time.monotonic()

        # Panel de metadata: se imprime UNA vez, antes de arrancar el Live,
        # y queda fijo arriba -- no forma parte de la región que Live
        # redibuja.
        lines = [
            f"Dataset: {meta.get('dataset_name', '?')}",
            f"Preprocessing: {meta.get('preprocessing_name', '?')}",
            f"Subjects: {meta.get('n_subjects', '?')}  ·  Sessions: {meta.get('n_sessions', '?')}"
            f"  ·  Runs pendientes: {total}",
            f"Modo: {meta.get('mode', '?')}",
        ]
        self.console.print(Panel("\n".join(lines), title="Barrido de preprocesamiento"))

        self._progress = Progress(
            TextColumn("Preprocessing"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            "subjects",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task("preprocessing", total=total)
        self._live = Live(self._progress, console=self.console, refresh_per_second=4)
        self._live.start()

    def on_run_start(self, partition: str) -> None:
        pass

    def on_run_result(self, partition: str, status: str) -> None:
        if status == "failed":
            self._failed.append(partition)
            # Imprimir sobre el mismo console que usa el Live: rich lo
            # inserta arriba de la región en vivo y no desaparece cuando
            # la barra avanza (mismo patrón que tqdm.write).
            self.console.print(f"[red]✗ {partition} — ver log para detalle[/red]")
        self._progress.advance(self._task_id)

    def on_finish(self, summary: dict) -> None:
        import datetime
        import time

        from rich.panel import Panel

        if self._live is not None:
            self._live.stop()

        elapsed = datetime.timedelta(seconds=round(time.monotonic() - self._start_time))
        lines = [
            f"Success: {summary.get('success', 0)}  ·  Failed: {summary.get('failed', 0)}",
            f"Tiempo total: {elapsed}",
        ]
        if self._failed:
            lines.append("Partitions fallidas:")
            lines.extend(f"  - {p}" for p in self._failed)
        self.console.print(Panel("\n".join(lines), title="Resumen del barrido"))
