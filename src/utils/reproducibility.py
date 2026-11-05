"""
Trío de reproducibilidad radical (PROTOCOL.md sección 8): config.json,
.git_commit, .docker_image. Compartido entre scripts/run_production.py
y src/preprocessing/database_preprocessor.py — no duplicar esta lógica.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

# Environment variable that gives the Docker image when --docker-image is not passed.
DOCKER_IMAGE_ENV_VAR = "JANUS_DOCKER_IMAGE"


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError: not inside a git repository.
        # FileNotFoundError: no git binary on the PATH.
        return "unknown-git-not-available"


def log_reproducibility_trio(run_dir: Path, config: dict, docker_image: Optional[str] = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / ".git_commit").write_text(get_git_commit_hash())
    # The flag wins over the environment variable. Without an image, a
    # .docker_image left by an earlier execution is removed, so the record
    # never names an image this execution did not use.
    docker_image = docker_image or os.environ.get(DOCKER_IMAGE_ENV_VAR)
    docker_image_path = run_dir / ".docker_image"
    if docker_image:
        docker_image_path.write_text(docker_image)
    else:
        docker_image_path.unlink(missing_ok=True)
