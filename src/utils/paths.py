"""
Punto único de acceso a rutas de JANUS-BCI.

Ningún otro módulo del repo debería leer os.environ directamente para
rutas: todo pasa por el objeto PATHS de este archivo. Si una ruta clave
no existe al arrancar, esto falla acá — rápido y con mensaje claro — en
vez de tirar un FileNotFoundError a mitad de un training de horas.
"""
from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JanusPaths(BaseSettings):
    "JANUS PATH Complete description later"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Debe existir SIEMPRE (es el propio repo corriendo) -> falla si no.
    janus_repo_root: DirectoryPath

    # Debe existir SIEMPRE (ahí vive la data cruda) -> falla si no.
    janus_data_root: DirectoryPath

    # Pueden no existir la primera vez -> se crean si hace falta.
    janus_results_root: Path
    janus_sandbox_root: Path

    janus_analytics_db: Optional[Path] = None

    @field_validator("janus_results_root", "janus_sandbox_root", mode="after")
    @classmethod
    def _ensure_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def analytics_db(self) -> Path:
        "complete the definition later"
        return self.janus_analytics_db or (self.janus_repo_root / "db" / "janus_analytics.db")

    @property
    def data_root(self) -> Path:
        "complete the definition later"
        return self.janus_data_root

    @property
    def results_root(self) -> Path:
        "complete the definition later"
        return self.janus_results_root

    @property
    def sandbox_root(self) -> Path:
        "complete the definition later"
        return self.janus_sandbox_root

    @property
    def preprocessed_root(self) -> Path:
        "complete the definition later"
        root = self.janus_data_root / "preprocessed"
        root.mkdir(parents=True, exist_ok=True)
        return root


# Se instancia una sola vez al importar el módulo. Cualquier .env mal
# configurado revienta acá, no en medio de un experimento.
PATHS = JanusPaths()


def experiment_path(
    results_root: Path, *,
    project_name: str, strategy_name: str, recipe_name: str,
    database_name: str, database_session: str, label: str,
) -> Path:
    """Single place that builds the experiment folder path:
    <results_root>/<project>/<strategy>/<recipe>/<database>_<session>/<label>.
    Pure: creates nothing and reads nothing."""
    return (
        Path(results_root) / project_name / strategy_name / recipe_name /
        f"{database_name}_{database_session}" / label
    )
