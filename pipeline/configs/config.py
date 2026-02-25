"""
Central configuration for the sustainability evaluation pipeline.

API keys and tokens are read from the environment so secrets stay out of source control.
Edit the repository lists below to change the study cohort.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import uuid


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_REPOS_DIR = DATA_DIR / "raw_repos"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_GIT_DIR = RESULTS_DIR / "git"
RESULTS_SONAR_DIR = RESULTS_DIR / "sonar"
RESULTS_LLM_DIR = RESULTS_DIR / "llm"
RESULTS_HOLISTIC_DIR = RESULTS_DIR / "holistic"
RESULTS_MERGED_DIR = RESULTS_DIR / "merged"
RESULTS_VALIDATION_DIR = RESULTS_DIR / "validation"

# SonarQube / SonarCloud
SONAR_HOST_URL = os.environ.get("SONAR_HOST_URL", "http://localhost:9000")
SONAR_TOKEN = os.environ.get("SONAR_TOKEN", "")
SONAR_SCANNER_BINARY = os.environ.get("SONAR_SCANNER", "sonar-scanner")

# LLM provider selection
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "local").strip().lower()

LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434").strip()

DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL

# Optional multi-model controls
LLM_MODEL_LIST = _split_csv(os.environ.get("LLM_MODEL_LIST", ""))
LLM_OUTPUT_PER_MODEL = os.environ.get("LLM_OUTPUT_PER_MODEL", "false").strip().lower() in {"1", "true", "yes"}

# Shared LLM limits
LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 400)
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 2)
LLM_TIMEOUT_S = _int_env("LLM_TIMEOUT_S", 300)

# LLM output controls
LLM_RESUME = os.environ.get("LLM_RESUME", "true").strip().lower() in {"1", "true", "yes"}
LLM_WRITE_EVERY = int(os.environ.get("LLM_WRITE_EVERY", "25"))
LLM_SORT_OUTPUT = os.environ.get("LLM_SORT_OUTPUT", "true").strip().lower() in {"1", "true", "yes"}

# LLM output paths
LEGACY_LLM_METRICS_PATH = RESULTS_DIR / "llm_metrics.csv"
LEGACY_HOLISTIC_ASSESSMENTS_PATH = RESULTS_DIR / "holistic_assessments.csv"


def _sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def _model_value(model: str | None = None) -> str:
    return (model or LLM_MODEL or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def _run_file_regex(prefix: str, safe_model: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(prefix)}_{re.escape(safe_model)}_run(\d+)\.csv$")


def _model_runs(directory: Path, prefix: str, safe_model: str) -> list[tuple[int, Path]]:
    regex = _run_file_regex(prefix, safe_model)
    runs: list[tuple[int, Path]] = []
    if not directory.exists():
        return runs
    for file in directory.glob(f"{prefix}_{safe_model}_run*.csv"):
        match = regex.match(file.name)
        if not match:
            continue
        run_idx = int(match.group(1))
        runs.append((run_idx, file))
    runs.sort(key=lambda item: item[0])
    return runs


def _latest_run_path(directory: Path, prefix: str, safe_model: str) -> Path | None:
    runs = _model_runs(directory, prefix, safe_model)
    if not runs:
        return None
    return runs[-1][1]


def _next_run_path(directory: Path, prefix: str, safe_model: str) -> Path:
    runs = _model_runs(directory, prefix, safe_model)
    next_run = runs[-1][0] + 1 if runs else 1
    return directory / f"{prefix}_{safe_model}_run{next_run:03d}.csv"


def llm_metrics_path(model: str | None = None, prefer_existing: bool = False) -> Path:
    model_value = (model or LLM_MODEL or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    safe_model = _sanitize_filename_component(model_value)
    path = _latest_run_path(RESULTS_LLM_DIR, "llm_metrics", safe_model) if prefer_existing else None
    if path is None and not prefer_existing:
        path = _next_run_path(RESULTS_LLM_DIR, "llm_metrics", safe_model)
    if path is None:
        path = RESULTS_LLM_DIR / f"llm_metrics_{safe_model}_run001.csv"
    if prefer_existing and not path.exists():
        if LEGACY_LLM_METRICS_PATH.exists():
            return LEGACY_LLM_METRICS_PATH
    return path


def holistic_assessments_path(model: str | None = None, prefer_existing: bool = False) -> Path:
    safe_model = _sanitize_filename_component(_model_value(model))
    path = _latest_run_path(RESULTS_HOLISTIC_DIR, "holistic_assessments", safe_model) if prefer_existing else None
    if path is None and not prefer_existing:
        path = _next_run_path(RESULTS_HOLISTIC_DIR, "holistic_assessments", safe_model)
    if path is None:
        path = RESULTS_HOLISTIC_DIR / f"holistic_assessments_{safe_model}_run001.csv"
    if prefer_existing and not path.exists():
        if LEGACY_HOLISTIC_ASSESSMENTS_PATH.exists():
            return LEGACY_HOLISTIC_ASSESSMENTS_PATH
    return path


def sonar_metrics_path() -> Path:
    return RESULTS_SONAR_DIR / "sonar_metrics.csv"


def git_metrics_path() -> Path:
    return RESULTS_GIT_DIR / "git_metrics.csv"


def final_dataset_path() -> Path:
    return RESULTS_MERGED_DIR / "final_dataset.csv"


def llm_repo_summary_path() -> Path:
    return RESULTS_LLM_DIR / "llm_repo_summary.csv"


def validate_report_path() -> Path:
    return RESULTS_VALIDATION_DIR / "validate_report.json"


def llm_parse_failures_path() -> Path:
    return RESULTS_LLM_DIR / "llm_parse_failures.jsonl"

# Validation controls
VALIDATE_OUTPUTS = os.environ.get("VALIDATE_OUTPUTS", "false").strip().lower() in {"1", "true", "yes"}

# Study cohort: list repository URLs for the dataset.
REPOSITORIES: tuple[str, ...] = (
    "https://github.com/django/django.git",
    "https://github.com/dropwizard/dropwizard.git",
    "https://github.com/psf/requests.git",
    "https://github.com/square/retrofit.git",
    "https://github.com/gin-gonic/gin.git",
    "https://github.com/expressjs/express.git",
    "https://github.com/nestjs/nest.git",
    "https://github.com/httpie/httpie.git",
    "https://github.com/axios/axios.git",
    "https://github.com/prometheus/prometheus.git",
)

# Flattened list for convenience.
ALL_REPOSITORIES: tuple[str, ...] = REPOSITORIES


def repo_dir_name(repo_url: str) -> str:
    """Return the directory name Git will create when cloning repo_url."""
    repo_name = repo_url.rstrip("/").split("/")[-1]
    return repo_name[:-4] if repo_name.endswith(".git") else repo_name


def ensure_data_dirs() -> None:
    """Create data directories if they do not already exist."""
    for path in (
        DATA_DIR,
        RAW_REPOS_DIR,
        RESULTS_DIR,
        RESULTS_GIT_DIR,
        RESULTS_SONAR_DIR,
        RESULTS_LLM_DIR,
        RESULTS_HOLISTIC_DIR,
        RESULTS_MERGED_DIR,
        RESULTS_VALIDATION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def git_head_sha(repo_path: Path) -> str | None:
    """Return the HEAD commit SHA for a repo, or None if unavailable."""
    if not repo_path or not repo_path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def short_run_id(length: int = 8) -> str:
    """Return a short unique identifier for output archival file names."""
    size = max(4, int(length))
    return uuid.uuid4().hex[:size]


def archive_existing_csv(path: Path, id_length: int = 8) -> Path | None:
    """
    Rename an existing CSV before a new run writes to the same path.

    The archived file is created in the same directory with format:
    <stem>_<shortid><suffix>
    """
    if not path.exists():
        return None

    for _ in range(25):
        archived = path.with_name(f"{path.stem}_{short_run_id(id_length)}{path.suffix}")
        if archived.exists():
            continue
        path.replace(archived)
        return archived

    raise RuntimeError(f"Failed to archive existing output for {path}")
