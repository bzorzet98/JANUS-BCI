"""
Orquestador único de preprocesamiento (preprocessing/DESIGN.md sección 4).
Reemplaza a `EEGDatabasePreprocessor`/`EEGDatabaseICAPreprocessor` de
repo_viejo — una forma nueva de preprocesar se agrega registrando un
stage handler (`EEGPreprocessor.register_stage_handler`) o un artifact
saver (`register_artifact_saver`), no subclaseando este orquestador.
"""
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from src.utils.json_utils import save_dict_as_json
from src.utils.paths import PATHS
from src.utils.progress_reporter import ProgressReporter
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, append_csv_row, diff_values

from .artifact_savers import ARTIFACT_SAVERS
from .preprocess_eeg import EEGPreprocessor, StageResult

logger = logging.getLogger(__name__)


def _save_processed_data(data: Any, base_path: Path, prefix: str) -> Optional[Path]:
    """Raw -> .fif, Epochs -> .epo.fif, dict -> .npy. Puerto verbatim de
    _save_results/_save_processed_data (idénticas en ambas clases viejas)."""
    base_path.mkdir(parents=True, exist_ok=True)
    if isinstance(data, mne.io.BaseRaw):
        out = base_path / f"{prefix}_raw.fif"
        data.save(out, overwrite=True, verbose=False)
        return out
    elif isinstance(data, mne.BaseEpochs):
        out = base_path / f"{prefix}_epo.fif"
        data.save(out, overwrite=True, verbose=False)
        return out
    elif isinstance(data, dict):
        last_out = None
        data_content = data.get("data")
        labels_content = data.get("labels")
        has_labels = isinstance(labels_content, np.ndarray) and len(labels_content) > 0
        if isinstance(data_content, np.ndarray) and has_labels:
            # data + labels planos: un solo .npz en vez de dos .npy sueltos.
            last_out = base_path / f"{prefix}.npz"
            np.savez(last_out, data=data_content, labels=labels_content)
            return last_out
        if isinstance(data_content, dict):
            for key, val in data_content.items():
                if isinstance(val, np.ndarray):
                    last_out = base_path / f"{prefix}_{key}.npy"
                    np.save(last_out, val)
        elif isinstance(data_content, np.ndarray):
            last_out = base_path / f"{prefix}_data.npy"
            np.save(last_out, data_content)
        if has_labels:
            last_out = base_path / f"{prefix}_labels.npy"
            np.save(last_out, labels_content)
        return last_out
    return None


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _diff_pipeline(old_pipeline: Optional[dict], new_pipeline: Optional[dict]) -> List[str]:
    """Compara dos `preprocessing_pipeline` (formato {'stages': [...]})
    stage por stage, identificando cada uno por `stage_name` para que el
    mensaje de error sea legible en vez de mostrar índices de lista."""
    old_stages = {
        s.get("stage_name", f"#{i}"): s
        for i, s in enumerate((old_pipeline or {}).get("stages", []))
    }
    new_stages = {
        s.get("stage_name", f"#{i}"): s
        for i, s in enumerate((new_pipeline or {}).get("stages", []))
    }

    diffs: List[str] = []
    for name in old_stages:
        if name not in new_stages:
            diffs.append(f"stage '{name}' eliminado")
    for name in new_stages:
        if name not in old_stages:
            diffs.append(f"stage '{name}' agregado")
    for name in old_stages:
        if name in new_stages and old_stages[name] != new_stages[name]:
            diffs.extend(diff_values(old_stages[name], new_stages[name], f"stage '{name}'"))
    return diffs


class EEGDatabasePreprocessor:
    def __init__(
        self,
        dataset,
        preprocessing_name: str,
        config: dict,
        docker_image: Optional[str] = None,
        reporter: Optional[ProgressReporter] = None,
        force: bool = False,
        overwrite_existing: bool = False,
    ):
        self.dataset = dataset
        self.preprocessing_name = preprocessing_name
        self.config = config
        self.docker_image = docker_image
        self.reporter = reporter or ProgressReporter()
        self.force = force
        self.overwrite_existing = overwrite_existing
        self.output_path = PATHS.preprocessed_root / preprocessing_name / dataset.code
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.worker = EEGPreprocessor(stages=config["preprocessing_pipeline"]["stages"])
        self._artifact_savers: Dict[str, Callable] = dict(ARTIFACT_SAVERS)
        self._done_partitions: set = set()
        self._metadata_enriched = False
        self.lifecycle = RunLifecycleManager(
            self.output_path,
            progress_filename="run_registry.csv",
            identity_filename="config.json",
            force=force,
            overwrite_existing=overwrite_existing,
            emit_running_row=False,
        )

    def register_artifact_saver(self, key: str, fn: Callable) -> None:
        self._artifact_savers[key] = fn

    def run(self) -> None:
        # Silencia el logging interno de MNE/mne_icalabel independientemente
        # de los `verbose` de cada config individual — una sola vez, acá,
        # porque este es el entrypoint real de un barrido (sandbox u otro
        # caller que use el orquestador directo lo hereda también).
        mne.set_log_level("WARNING")

        # Antes de tocar nada en disco: si ya hay una corrida previa con un
        # preprocessing_pipeline distinto para este preprocessing_name, no
        # seguimos en silencio (o pedimos --overwrite-existing, o
        # preguntamos por TTY, o rechazamos).
        self.lifecycle.check_identity_or_raise(
            new_config=self.config,
            extract_identity=lambda c: c.get("preprocessing_pipeline"),
            diff_fn=_diff_pipeline,
            extra_cleanup=lambda: [p.unlink() for p in self.output_path.glob("*.csv")],
            context_label=f"config.json (preprocessing_name='{self.preprocessing_name}')",
        )

        # Trío de reproducibilidad + manifest, una sola vez, antes del loop
        # (DESIGN.md sección 4 punto 5).
        log_reproducibility_trio(self.output_path, self.config, self.docker_image)
        self._write_dataset_description()

        subjects_list = self.dataset.get_subjects_list()
        sessions = getattr(self.dataset, "sessions", ["session_1"])

        # Único chequeo de particiones ya resueltas, antes de cargar
        # ningún subject: se predice la lista completa sin I/O
        # (_build_partition_list) y se descarta lo ya hecho
        # (lifecycle.filter_pending) para saber, de antemano, qué (subject,
        # session) hace falta cargar de verdad.
        self._done_partitions = set() if self.force else self._load_done_partitions()
        all_partitions = self._build_partition_list()
        pending = all_partitions if self.force else self.lifecycle.filter_pending(
            all_partitions, self._done_partitions, key=lambda item: item[2]
        )

        if not pending:
            logger.info(
                "Todo ya procesado (%d combinaciones subject×session×context "
                "predichas) -- nada para hacer.", len(all_partitions),
            )
            return

        if self._done_partitions:
            logger.info(
                "Retomando barrido: %d combinaciones ya resueltas se van a omitir "
                "(--force para ignorar el registro y reprocesar todo).",
                len(self._done_partitions),
            )

        pending_subject_sessions = sorted(set((subject, session) for subject, session, _ in pending))
        logger.info(
            "Starting preprocessing: %d subject×session a cargar (de %d totales).",
            len(pending_subject_sessions), len(subjects_list) * len(sessions),
        )
        self.reporter.on_start(len(pending), meta={
            "dataset_name": getattr(self.dataset, "code", type(self.dataset).__name__),
            "preprocessing_name": self.preprocessing_name,
            "n_subjects": len(subjects_list),
            "n_sessions": len(sessions),
            "mode": "docker" if self.docker_image else "local",
        })

        for subject, session in pending_subject_sessions:
            try:
                subject_obj = self.dataset.get_subject(subject, session=session)
            except Exception as e:
                logger.error("CRITICAL ERROR loading subject %s: %s", subject, e)
                partition = f"subject_{subject:02d}/unknown/unknown"
                self.reporter.on_run_start(partition)
                self._append_run_registry(
                    partition=partition,
                    subject=subject,
                    session=session,
                    context=None,
                    run_id=None,
                    status="failed",
                    ts_start=_now(),
                    ts_end=_now(),
                    output_path=None,
                )
                self.reporter.on_run_result(partition, "failed")
                continue
            if subject_obj is None:
                continue
            for context, run_id, raw_data in subject_obj.iter_all_runs():
                partition = f"subject_{subject:02d}/{context}/{run_id}"
                if partition in self._done_partitions:
                    logger.info("Omitiendo %s (ya resuelto en run_registry.csv)", partition)
                    continue
                self._process_one_run(subject, session, context, run_id, raw_data)

        logger.info("--- Preprocessing Finished ---")
        self.reporter.on_finish(self._build_summary())

    def _process_one_run(self, subject, session, context, run_id, raw_data) -> None:
        partition = f"subject_{subject:02d}/{context}/{run_id}"
        prefix = f"subject_{subject:02d}_{context}_{run_id}"
        ts_start = _now()
        status = "failed"
        output_path = None
        self.reporter.on_run_start(partition)
        try:
            logger.info("Processing %s", partition)
            result = self.worker.process(raw_data)
            if not self._metadata_enriched:
                # Only the first successful run is needed: every run goes
                # through the same stage pipeline, so the final data type
                # (and therefore which fields are readable) is the same
                # for the whole sweep.
                self._enrich_dataset_description(result.data)
                self._metadata_enriched = True
            data_dir = self.output_path / session / f"subject_{subject:02d}"
            output_path = _save_processed_data(result.data, data_dir, prefix)
            self._save_artifacts(result, subject, prefix)
            self._append_stage_metrics(result.metrics, partition)
            self._append_detail_tables(result.detail_tables, partition)
            status = "success"
        except Exception as e:
            logger.error("ERROR processing %s: %s", partition, e)
        finally:
            ts_end = _now()
            self._append_run_registry(
                partition=partition,
                subject=subject,
                session=session,
                context=context,
                run_id=run_id,
                status=status,
                ts_start=ts_start,
                ts_end=ts_end,
                output_path=str(output_path) if output_path else None,
            )
            self.reporter.on_run_result(partition, status)

    def _save_artifacts(self, result: StageResult, subject, prefix: str) -> None:
        # Un saver que falla (ej. no puede graficar por falta de montage)
        # no debe tirar abajo el run entero ni perder metrics/detail_tables
        # ya calculados — mismo espíritu que "sin saver registrado, se
        # omite" (DESIGN.md sección 4 punto 3): nunca rompe el barrido.
        for stage_name, stage_artifacts in result.artifacts.items():
            stage_detail_tables = result.detail_tables.get(stage_name, {})
            for key, value in stage_artifacts.items():
                saver = self._artifact_savers.get(key)
                if saver is None:
                    logger.info(
                        "sin saver registrado para artifact '%s' del stage '%s', se omite",
                        key, stage_name,
                    )
                    continue
                output_dir = self.output_path / "artifacts" / stage_name / f"subject_{subject:02d}"
                try:
                    saver(
                        value,
                        artifacts=stage_artifacts,
                        detail_tables=stage_detail_tables,
                        output_dir=output_dir,
                        prefix=prefix,
                    )
                except Exception as e:
                    logger.warning(
                        "saver de artifact '%s' del stage '%s' falló: %s", key, stage_name, e,
                    )

    def _build_partition_list(self) -> List[Tuple[int, str, str]]:
        """Predicted (subject, session, partition) triples from
        subject_list x sessions x data_to_load -- no I/O, no
        get_subject() calls, instant even for a sweep of thousands of
        subjects. This is an approximation: it assumes one run_id
        ("run_1") per data_to_load entry, which matches Cho2017's
        motor_imagery/movement and Lee2019's motor_imagery, but not
        Cho2017's 'rest' (real key: 'recording') or 'noise' (5 real
        sub-keys) contexts, nor Dreyer2023's runs (existence depends on
        which .gdf files are actually on disk per subject, only knowable
        inside get_subject()). This never causes a real pending run to be
        silently skipped -- the per-run check against run_registry.csv
        inside the main loop (self._done_partitions) stays authoritative
        and catches anything this prediction gets wrong. It only means
        the subject-level pre-filter below is a weaker optimization for
        those datasets: get_subject() may still run for a subject that's
        actually fully done, because none of its real partitions happen
        to match a predicted one."""
        subjects_list = self.dataset.get_subjects_list()
        sessions = getattr(self.dataset, "sessions", ["session_1"])
        contexts = getattr(self.dataset, "data_to_load", None) or self.dataset.get_data_available()
        return [
            (subject, session, f"subject_{subject:02d}/{context}/run_1")
            for subject in subjects_list
            for session in sessions
            for context in contexts
        ]

    def _load_done_partitions(self) -> set:
        """Partitions con status == 'success' en run_registry.csv que
        además tienen su archivo de salida presente en disco — se saltean
        en este barrido. Delegado a RunLifecycleManager.load_success_partitions;
        la columna `output_path` (dónde vive el criterio de "¿sigue en
        disco?") es un detalle específico de este CSV, así que se resuelve
        acá y no en el componente compartido."""
        registry_path = self.output_path / "run_registry.csv"
        if not registry_path.exists():
            return set()
        registry = pd.read_csv(registry_path)
        if registry.empty:
            return set()
        latest = registry.drop_duplicates(subset="partition", keep="last")
        output_by_partition = latest.set_index("partition")["output_path"].to_dict()

        def _still_present(partition: str) -> bool:
            output_path = output_by_partition.get(partition)
            if output_path is None or (isinstance(output_path, float) and pd.isna(output_path)):
                return False
            return Path(str(output_path)).exists()

        return self.lifecycle.load_success_partitions(still_present=_still_present)

    def _build_summary(self) -> Dict[str, Any]:
        # Se lee de vuelta run_registry.csv/stage_metrics.csv en vez de
        # acumular en memoria durante el barrido (DESIGN.md sección 4
        # punto 4 pide no acumular, así un barrido cortado a mitad de
        # camino deja los CSV consultables igual).
        summary: Dict[str, Any] = {"total": 0, "success": 0, "failed": 0}
        registry_path = self.output_path / "run_registry.csv"
        if registry_path.exists():
            registry = pd.read_csv(registry_path)
            summary["total"] = len(registry)
            status_counts = registry["status"].value_counts()
            summary["success"] = int(status_counts.get("success", 0))
            summary["failed"] = int(status_counts.get("failed", 0))
        metrics_path = self.output_path / "stage_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            for (stage_name, metric_name), values in metrics.groupby(["stage_name", "metric_name"])["value"]:
                numeric_values = pd.to_numeric(values, errors="coerce").dropna()
                if len(numeric_values) > 0:
                    summary[f"{stage_name}.{metric_name} (avg)"] = round(numeric_values.mean(), 4)
        return summary

    def _append_stage_metrics(self, metrics: Dict[str, dict], partition: str) -> None:
        path = self.output_path / "stage_metrics.csv"
        for stage_name, stage_metrics in metrics.items():
            for metric_name, value in stage_metrics.items():
                append_csv_row(path, {
                    "partition": partition,
                    "stage_name": stage_name,
                    "metric_name": metric_name,
                    "value": value,
                })

    def _append_detail_tables(self, detail_tables: Dict[str, dict], partition: str) -> None:
        for stage_name, stage_details in detail_tables.items():
            for detail_key, rows in stage_details.items():
                path = self.output_path / f"{stage_name}_{detail_key}.csv"
                for row in rows:
                    append_csv_row(path, {"partition": partition, **row})

    def _append_run_registry(
        self, *, partition, subject, session, context, run_id, status, ts_start, ts_end, output_path
    ) -> None:
        append_csv_row(self.output_path / "run_registry.csv", {
            "partition": partition,
            "subject_id": subject,
            "session": session,
            "context": context,
            "run_id": run_id,
            "status": status,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "output_path": output_path,
        })

    def _write_dataset_description(self) -> None:
        info = {
            "dataset_code": getattr(self.dataset, "code", "Unknown"),
            "subject_list": self.dataset.get_subjects_list(),
            "sessions_availables": getattr(self.dataset, "sessions", ["session_1"]),
            # Fix de la deriva DESIGN.md sección 1: siempre lo que
            # realmente se cargó (y por lo tanto quedó escrito a disco),
            # nunca la lista fija de contextos posibles del dataset.
            "data_availables": getattr(self.dataset, "data_to_load", None) or self.dataset.get_data_available(),
            "sfreq": getattr(self.dataset, "sfreq", None),
            "ch_names": getattr(self.dataset, "ch_names", []),
            "ch_types": getattr(self.dataset, "ch_types", []),
            "event_id": getattr(self.dataset, "event_id", {}),
            "standard_montage": getattr(self.dataset, "standard_montage", None),
            "subjects_individual_metadata": {},
        }
        subjects_metadata = getattr(self.dataset, "subjects_metadata", None)
        if subjects_metadata is not None:
            for _, row in subjects_metadata.iterrows():
                sub_id = row.get("subject_id", row.get("id", row.get("Subject")))
                if sub_id is not None:
                    info["subjects_individual_metadata"][int(sub_id)] = row.to_dict()
        save_dict_as_json(str(self.output_path), info, "dataset_description.json")

    @staticmethod
    def _extract_epoch_metadata(data: Any) -> Dict[str, Any]:
        """Read sfreq/n_times/tmin from an actual processed result, never
        declared or computed from config. Returns an empty dict (no field
        added, no default guessed) unless `data` is an mne.Epochs or the
        dict produced by extract_data_from_epochs -- e.g. plain Raw (no
        epoching stage) yields {}."""
        if isinstance(data, mne.BaseEpochs):
            return {"sfreq": data.info["sfreq"], "n_times": len(data.times), "tmin": data.tmin}
        if isinstance(data, dict) and isinstance(data.get("data"), np.ndarray):
            metadata: Dict[str, Any] = {"n_times": data["data"].shape[-1]}
            if "sfreq" in data:
                metadata["sfreq"] = data["sfreq"]
            if "tmin" in data:
                metadata["tmin"] = data["tmin"]
            if "tmax" in data:
                metadata["tmax"] = data["tmax"]
            return metadata
        return {}

    def _enrich_dataset_description(self, data: Any) -> None:
        """Patch dataset_description.json with sfreq/n_times/tmin read
        from the first successful run's real result, once, overriding the
        declared-upfront sfreq with the real one. No-op if `data` isn't
        Epochs-shaped (Raw-only pipelines keep the declared sfreq and
        never get n_times/tmin)."""
        metadata = self._extract_epoch_metadata(data)
        if not metadata:
            return
        description_path = self.output_path / "dataset_description.json"
        info = json.loads(description_path.read_text())
        info.update(metadata)
        save_dict_as_json(str(self.output_path), info, "dataset_description.json")
