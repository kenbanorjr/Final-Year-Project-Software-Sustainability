"""
Baseline SonarQube pipeline.

Runs sonar-scanner for each cloned repository and pulls file-level metrics via
the SonarQube Web API.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

import config

SONAR_METRICS = (
    "complexity",
    "sqale_index",
    "violations",
    "code_smells",
    "bugs",
    "vulnerabilities",
    "duplicated_blocks",
    "duplicated_lines_density",
    "test_success_density",
)

# Folders Sonar expects for Java bytecode (created if missing).
JAVA_BIN_DIRS = (
    "target/classes",
    "build/classes",
    "build/classes/java/main",
)


def project_key_for_repo(repo_path: Path) -> str:
    """Derive a stable Sonar project key from the directory name."""
    return repo_path.name.replace("-", "_").replace(" ", "_")


def _has_java_files(repo_path: Path) -> bool:
    """Return True if the repo contains any .java files."""
    for path in repo_path.rglob("*.java"):
        if path.is_file():
            return True
    return False


def run_sonar_scan(repo_path: Path, project_key: str) -> None:
    """Invoke sonar-scanner for the given repository."""
    env = os.environ.copy()
    env.setdefault("SONAR_HOST_URL", config.SONAR_HOST_URL)
    if config.SONAR_TOKEN:
        env["SONAR_TOKEN"] = config.SONAR_TOKEN

    cmd = [
        config.SONAR_SCANNER_BINARY,
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.projectName={repo_path.name}",
        f"-Dsonar.sources=.",
        f"-Dsonar.host.url={config.SONAR_HOST_URL}",
        "-Dsonar.sourceEncoding=UTF-8",
    ]

    if _has_java_files(repo_path):
        # Ensure expected binary directories exist to satisfy Java analyzer requirements.
        for rel_dir in JAVA_BIN_DIRS:
            (repo_path / rel_dir).mkdir(parents=True, exist_ok=True)
        cmd.append(f"-Dsonar.java.binaries={','.join(JAVA_BIN_DIRS)}")

    printable = " ".join(cmd)
    print(f"[sonar] {repo_path.name}: {printable}")
    subprocess.run(cmd, cwd=repo_path, env=env, check=True)


def _request_session() -> requests.Session:
    session = requests.Session()
    if config.SONAR_TOKEN:
        session.auth = (config.SONAR_TOKEN, "")
    return session


def fetch_file_metrics(project_key: str) -> List[dict]:
    """Fetch file-level metrics from SonarQube for a project."""
    session = _request_session()
    results: List[dict] = []
    page = 1
    page_size = 500
    total_pages = 1

    while page <= total_pages:
        resp = session.get(
            f"{config.SONAR_HOST_URL}/api/measures/component_tree",
            params={
                "component": project_key,
                "metricKeys": ",".join(SONAR_METRICS),
                "qualifiers": "FIL",
                "p": page,
                "ps": page_size,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        components = payload.get("components", [])
        paging = payload.get("paging", {}) or {}
        total = paging.get("total", len(components))
        page_size = paging.get("pageSize", page_size)
        total_pages = max(1, math.ceil(total / page_size))

        for comp in components:
            path = comp.get("path") or comp.get("name") or ""
            measures = {m["metric"]: m.get("value") for m in comp.get("measures", [])}
            results.append(
                {
                    "project_key": project_key,
                    "file_path": path.replace("\\", "/"),
                    "sonar_complexity": measures.get("complexity"),
                    "sonar_sqale_index": measures.get("sqale_index"),
                    "sonar_violations": measures.get("violations"),
                    "sonar_code_smells": measures.get("code_smells"),
                    "sonar_bugs": measures.get("bugs"),
                    "sonar_vulnerabilities": measures.get("vulnerabilities"),
                    "sonar_duplicated_blocks": measures.get("duplicated_blocks"),
                    "sonar_duplicated_lines_density": measures.get("duplicated_lines_density"),
                    "sonar_test_success_density": measures.get("test_success_density"),
                }
            )

        page += 1

    return results


def collect_sonar_metrics(repo_paths: Iterable[Path], output_path: Path | None = None) -> Path:
    """Run scans and collect per-file metrics for the provided repositories."""
    output = output_path or (config.RESULTS_DIR / "sonar_metrics.csv")
    all_rows: List[dict] = []

    for repo_path in repo_paths:
        if not repo_path.exists():
            print(f"[sonar] Skipping missing repository at {repo_path}")
            continue

        project_key = project_key_for_repo(repo_path)
        try:
            run_sonar_scan(repo_path, project_key)
            print(f"[sonar] Fetching metrics for {project_key}")
            rows = fetch_file_metrics(project_key)
            for row in rows:
                row["repo"] = repo_path.name
                all_rows.append(row)
        except Exception as exc:
            print(f"[sonar] Failed for {repo_path.name}: {exc}")
            continue

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.sort_values(["repo", "file_path"], inplace=True)
    df.to_csv(output, index=False)
    print(f"[sonar] Wrote SonarQube metrics to {output}")
    return output


def main() -> None:
    repo_paths = [config.RAW_REPOS_DIR / config.repo_dir_name(url) for url in config.ALL_REPOSITORIES]
    collect_sonar_metrics(repo_paths)


if __name__ == "__main__":
    main()
