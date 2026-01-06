"""
Git mining engine for the sustainability study.

Responsibilities:
- Clone repositories defined in configs/config.py.
- Identify representative files (Python/Java, 50–400 LOC).
- Compute churn and bus-factor indicators over the last 12 months.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pydriller import Repository

from configs import config

REPRESENTATIVE_EXTENSIONS = {".py", ".java", ".go", ".js", ".ts"}
EXTENSION_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
}
MIN_LOC = 50
MAX_LOC = 400
WINDOW_DAYS = 365  # 12 months


def clone_repositories(repo_urls: Iterable[str], base_dir: Path | None = None) -> None:
    """Clone repositories if they do not already exist."""
    config.ensure_data_dirs()
    destination = base_dir or config.RAW_REPOS_DIR
    destination.mkdir(parents=True, exist_ok=True)

    for repo_url in repo_urls:
        target_dir = destination / config.repo_dir_name(repo_url)
        if target_dir.exists():
            print(f"[clone] Skipping {repo_url} (already exists at {target_dir})")
            continue

        cmd = ["git", "clone", repo_url, str(target_dir)]
        print(f"[clone] Cloning {repo_url} into {target_dir}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[clone] Failed to clone {repo_url} (exit code {exc.returncode})")


def _count_loc(path: Path) -> int:
    """Count lines up to MAX_LOC+1 so we can drop very large files early."""
    count = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for count, _line in enumerate(handle, start=1):
            if count > MAX_LOC:
                break
    return count


def _infer_language(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return EXTENSION_LANGUAGE.get(suffix, "other")


def find_representative_files(repo_path: Path) -> Dict[str, int]:
    """Return mapping of relative file path -> LOC for representative files."""
    reps: Dict[str, int] = {}
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in REPRESENTATIVE_EXTENSIONS:
            continue
        loc = _count_loc(file_path)
        if MIN_LOC <= loc <= MAX_LOC:
            rel_path = file_path.relative_to(repo_path).as_posix()
            reps[rel_path] = loc
    return reps


def _repo_url_lookup() -> Dict[str, str]:
    """Map cloned directory name to repository URL."""
    return {config.repo_dir_name(url): url for url in config.ALL_REPOSITORIES}


def _line_changes(mod) -> Tuple[int, int]:
    """Return (added, deleted) with compatibility across pydriller versions."""
    added = (
        getattr(mod, "added", None)
        or getattr(mod, "added_lines", None)
        or getattr(mod, "additions", None)
        or 0
    )
    deleted = (
        getattr(mod, "removed", None)
        or getattr(mod, "deleted_lines", None)
        or getattr(mod, "deletions", None)
        or 0
    )
    return int(added), int(deleted)


def mine_repository(repo_path: Path, since: datetime) -> List[dict]:
    """
    Traverse commits in the last WINDOW_DAYS to compute churn and bus-factor risk.

    Churn: total added + removed lines per file.
    Bus-factor risk: file touched by exactly one author in the window.
    """
    analysis_date_utc = config.now_utc_iso()
    repo_head_sha = config.git_head_sha(repo_path)
    representative_files = find_representative_files(repo_path)
    if not representative_files:
        return []

    author_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    author_line_additions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    churn: Dict[str, int] = defaultdict(int)
    added_totals: Dict[str, int] = defaultdict(int)
    deleted_totals: Dict[str, int] = defaultdict(int)
    repo_commit_count = 0
    repo_authors: set[str] = set()

    # Some pydriller versions do not support filepath_regex; filter manually instead.
    repo_mining = Repository(str(repo_path), since=since)
    for commit in repo_mining.traverse_commits():
        author = commit.author.name or commit.author.email or "unknown"
        repo_commit_count += 1
        repo_authors.add(author)

        mods = getattr(commit, "modifications", None)
        if mods is None:
            mods = getattr(commit, "modified_files", [])
        for mod in mods:
            path = mod.new_path or mod.old_path
            if not path:
                continue
            rel_path = Path(path).as_posix()
            if rel_path not in representative_files:
                continue

            added, deleted = _line_changes(mod)
            churn[rel_path] += added + deleted
            added_totals[rel_path] += added
            deleted_totals[rel_path] += deleted
            author_counts[rel_path][author] += 1
            author_line_additions[rel_path][author] += added

    repo_url = _repo_url_lookup().get(repo_path.name, "")
    seed_category = config.repo_category(repo_url)
    activity_label = (
        "ACTIVE"
        if repo_commit_count >= 24 and len(repo_authors) >= 3
        else "STAGNANT"
    )

    rows: List[dict] = []
    for rel_path, loc in representative_files.items():
        authors_for_file = author_counts.get(rel_path, {})
        unique_authors = len(authors_for_file)
        total_modifications = sum(authors_for_file.values())
        bus_factor_single_dev = unique_authors == 1 and total_modifications > 0

        line_contribs = author_line_additions.get(rel_path, {})
        total_added_lines = sum(line_contribs.values())

        dominant_author = ""
        dominant_author_share = 0.0

        if total_added_lines > 0:
            dominant_author, max_lines = max(line_contribs.items(), key=lambda kv: kv[1])
            dominant_author_share = max_lines / total_added_lines if total_added_lines else 0.0
        elif total_modifications > 0:
            # Fallback to modification counts when we lack line additions.
            dominant_author, max_mods = max(authors_for_file.items(), key=lambda kv: kv[1])
            dominant_author_share = max_mods / total_modifications

        bus_factor_75_dominant = dominant_author_share >= 0.75 and (total_added_lines > 0 or total_modifications > 0)

        rows.append(
            {
                "repo": repo_path.name,
                "repo_url": repo_url,
                "seed_category": seed_category,
                "activity_label": activity_label,
                "analysis_date_utc": analysis_date_utc,
                "repo_head_sha": repo_head_sha,
                "file_path": rel_path,
                "file_language": _infer_language(rel_path),
                "absolute_path": str((repo_path / rel_path).resolve()),
                "lines_of_code": loc,
                "unique_authors_12m": unique_authors,
                "repo_unique_authors_12m": len(repo_authors),
                "repo_commits_12m": repo_commit_count,
                "repo_commits_per_month_12m": repo_commit_count / 12.0,
                "churn_12m": churn.get(rel_path, 0),
                "added_lines_12m": added_totals.get(rel_path, 0),
                "deleted_lines_12m": deleted_totals.get(rel_path, 0),
                "bus_factor_single_dev": bus_factor_single_dev,
                "bus_factor_75_dominant_author": bus_factor_75_dominant,
                "dominant_author": dominant_author,
                "dominant_author_share": dominant_author_share,
                "last_12m_observed_commits": total_modifications,
            }
        )

    return rows


def run_git_mining(output_path: Path | None = None) -> Path:
    """Mine all configured repositories and persist git metrics CSV."""
    output = output_path or (config.RESULTS_DIR / "git_metrics.csv")
    config.ensure_data_dirs()

    since = datetime.utcnow() - timedelta(days=WINDOW_DAYS)
    all_rows: List[dict] = []

    for repo_url in config.ALL_REPOSITORIES:
        repo_path = config.RAW_REPOS_DIR / config.repo_dir_name(repo_url)
        if not repo_path.exists():
            print(f"[mine] Missing repository at {repo_path}; skipping metrics.")
            continue

        print(f"[mine] Mining git history for {repo_path.name}")
        rows = mine_repository(repo_path, since)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.sort_values(["repo", "file_path"], inplace=True)
    df.to_csv(output, index=False)
    print(f"[mine] Wrote git metrics to {output}")
    return output


def main() -> None:
    clone_repositories(config.ALL_REPOSITORIES)
    run_git_mining()


if __name__ == "__main__":
    main()
