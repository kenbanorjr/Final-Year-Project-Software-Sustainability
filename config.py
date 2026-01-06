"""
Central configuration for the sustainability evaluation pipeline.

API keys and tokens are read from the environment so secrets stay out of source control.
Edit the repository lists below to change the study cohort.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_REPOS_DIR = DATA_DIR / "raw_repos"
RESULTS_DIR = DATA_DIR / "results"

# SonarQube / SonarCloud
SONAR_HOST_URL = os.environ.get("SONAR_HOST_URL", "http://localhost:9000")
SONAR_TOKEN = os.environ.get("SONAR_TOKEN", "")
SONAR_SCANNER_BINARY = os.environ.get("SONAR_SCANNER", "sonar-scanner")

# LLM provider selection
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

# OpenAI-compatible settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# Ollama settings (OpenAI-compatible local server)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")

# Gemini settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-pro")

# Shared LLM limits
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "400"))

# LLM sampling (set size > 0 to sample representative files for LLM judging)
LLM_SAMPLE_SIZE = int(os.environ.get("LLM_SAMPLE_SIZE", "0"))
LLM_SAMPLE_SEED = int(os.environ.get("LLM_SAMPLE_SEED", "42"))
LLM_SAMPLE_STRATEGY = os.environ.get("LLM_SAMPLE_STRATEGY", "stratified").lower()
LLM_SAMPLE_MIN_PER_REPO = int(os.environ.get("LLM_SAMPLE_MIN_PER_REPO", "0"))
LLM_SAMPLE_RISK_FRACTION = float(os.environ.get("LLM_SAMPLE_RISK_FRACTION", "0.35"))
LLM_SAMPLE_RISK_CHURN_QUANTILE = float(os.environ.get("LLM_SAMPLE_RISK_CHURN_QUANTILE", "0.75"))
LLM_SAMPLE_MANIFEST = os.environ.get("LLM_SAMPLE_MANIFEST", "")

# LLM output controls
LLM_RESUME = os.environ.get("LLM_RESUME", "true").lower() in {"1", "true", "yes"}
LLM_WRITE_EVERY = int(os.environ.get("LLM_WRITE_EVERY", "25"))
LLM_SORT_OUTPUT = os.environ.get("LLM_SORT_OUTPUT", "true").lower() in {"1", "true", "yes"}

# Validation controls
VALIDATE_OUTPUTS = os.environ.get("VALIDATE_OUTPUTS", "false").lower() in {"1", "true", "yes"}

# Study cohort: tune ACTIVE/STAGNANT to shape the dataset.
REPOSITORIES: dict[str, tuple[str, ...]] = {
    "ACTIVE": (
        "https://github.com/spring-projects/spring-petclinic",
        "https://github.com/axios/axios",
        "https://github.com/gin-gonic/gin",
        "https://github.com/google/guava",
        "https://github.com/pallets/flask.git",
    ),
    "STAGNANT": (
        "https://github.com/iluwatar/java-design-patterns",
        "https://github.com/apache/commons-lang",
        "https://github.com/junit-team/junit5",
        "https://github.com/checkstyle/checkstyle",
        "https://github.com/ReactiveX/RxJava",
        "https://github.com/dropwizard/dropwizard",
        "https://github.com/mybatis/mybatis-3",
        "https://github.com/google/subpar",
    ),
}

# Flattened list for convenience.
ALL_REPOSITORIES: tuple[str, ...] = tuple(
    repo_url for group in REPOSITORIES.values() for repo_url in group
)


def repo_dir_name(repo_url: str) -> str:
    """Return the directory name Git will create when cloning repo_url."""
    repo_name = repo_url.rstrip("/").split("/")[-1]
    return repo_name[:-4] if repo_name.endswith(".git") else repo_name


def repo_category(repo_url: str) -> str:
    """Return ACTIVE/STAGNANT label for a repository URL."""
    for category, urls in REPOSITORIES.items():
        if repo_url in urls:
            return category
    return "UNKNOWN"


def ensure_data_dirs() -> None:
    """Create data directories if they do not already exist."""
    for path in (DATA_DIR, RAW_REPOS_DIR, RESULTS_DIR):
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
