"""
Git mining engine for the sustainability study.

Responsibilities:
- Clone repositories defined in configs/config.py.
- Identify files for git mining (Sonar file list when available; otherwise
  representative files: Python/Java/Kotlin/Go/JS/TS).
- Compute churn and knowledge-concentration indicators over the last 12 months.
- Compute file-level metrics: recency_days, file_age_days, commit_count_12m, bus_factor_estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pydriller import Repository

from pipeline.configs import config
from pipeline.configs.general_repo_filter import is_excluded_path, is_included_path

AUTHOR_ALIASES = {
    # "alias@example.com": "canonical@example.com",
    # "alias name": "canonical name",
}
EXTENSION_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
}
WINDOW_DAYS = 365  # 12 months
MINER_CACHE_VERSION = "v1"


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
    """Count lines in a file."""
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _line in handle)


def _infer_language(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return EXTENSION_LANGUAGE.get(suffix, "other")


def _normalize_author(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _author_display(author) -> str:
    name = getattr(author, "name", None)
    if isinstance(name, str):
        name = name.strip()
    email = getattr(author, "email", None)
    if isinstance(email, str):
        email = email.strip()
    return name or email or "unknown"


def _normalized_author_aliases() -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for alias, canonical in AUTHOR_ALIASES.items():
        alias_key = _normalize_author(alias)
        canonical_key = _normalize_author(canonical)
        if alias_key and canonical_key:
            normalized[alias_key] = canonical_key
    return normalized


_AUTHOR_ALIAS_MAP = _normalized_author_aliases()


def _apply_author_alias(value: str) -> str:
    return _AUTHOR_ALIAS_MAP.get(value, value)


def _author_key(author) -> str:
    email = _normalize_author(getattr(author, "email", None))
    name = _normalize_author(getattr(author, "name", None))
    if email:
        email = _apply_author_alias(email)
    if name:
        name = _apply_author_alias(name)
    if email:
        return email
    if name:
        return name
    return "unknown"


def _canonical_path(path: str, aliases: Dict[str, str]) -> str:
    current = Path(path).as_posix()
    seen = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


def _approx_truck_factor(author_counts: Dict[str, Dict[str, int]]) -> int | None:
    file_to_authors: Dict[str, set[str]] = {}
    for file_path, authors in author_counts.items():
        if authors:
            file_to_authors[file_path] = set(authors.keys())
    if not file_to_authors:
        return None

    all_authors: set[str] = set()
    for authors in file_to_authors.values():
        all_authors.update(authors)
    if not all_authors:
        return None

    author_file_counts = {
        author: sum(1 for authors in file_to_authors.values() if author in authors)
        for author in all_authors
    }
    ranked_authors = sorted(
        all_authors,
        key=lambda author: author_file_counts.get(author, 0),
        reverse=True,
    )

    def coverage(remaining: set[str]) -> float:
        covered = sum(1 for authors in file_to_authors.values() if authors & remaining)
        return covered / len(file_to_authors)

    remaining = set(all_authors)
    removed = 0
    for author in ranked_authors:
        remaining.discard(author)
        removed += 1
        if coverage(remaining) < 0.5:
            return removed
    return removed


def _compute_file_bus_factor(author_commit_counts: Dict[str, int]) -> int | None:
    """
    Compute file-level bus factor as minimum k such that top-k authors
    account for >= 50% of commits touching this file.
    
    Returns None if no commits found.
    """
    if not author_commit_counts:
        return None
    
    total_commits = sum(author_commit_counts.values())
    if total_commits == 0:
        return None
    
    # Sort authors by commit count descending
    sorted_authors = sorted(author_commit_counts.items(), key=lambda x: x[1], reverse=True)
    
    cumulative = 0
    for k, (author, count) in enumerate(sorted_authors, start=1):
        cumulative += count
        if cumulative / total_commits >= 0.5:
            return k
    
    # All authors needed
    return len(sorted_authors)


def _normalize_path(path: str) -> str:
    """Normalize path: forward slashes, strip leading ./ and trailing /."""
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    # Strip leading ./
    while normalized.startswith("./"):
        normalized = normalized[2:]
    # Strip leading /
    normalized = normalized.lstrip("/")
    # Strip trailing /
    normalized = normalized.rstrip("/")
    return normalized


def _target_files_hash(target_files: Dict[str, int]) -> str:
    """Stable hash of target file paths for cache invalidation."""
    digest = hashlib.sha256()
    for rel_path in sorted(target_files.keys()):
        digest.update(rel_path.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def _miner_cache_dir() -> Path:
    return config.RESULTS_GIT_DIR / "cache"


def _miner_cache_path(
    repo_name: str,
    repo_head_sha: str,
    since_utc_date: str,
    target_files: Dict[str, int],
) -> Path:
    """Build cache path for a repo mining result."""
    safe_repo = repo_name.replace("/", "_")
    head = (repo_head_sha or "unknown").strip() or "unknown"
    head_short = head[:12]
    key_parts = f"{safe_repo}|{head}|{WINDOW_DAYS}|{since_utc_date}|{_target_files_hash(target_files)}|{MINER_CACHE_VERSION}"
    key_hash = hashlib.sha256(key_parts.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return _miner_cache_dir() / f"{safe_repo}_{head_short}_{key_hash}.json"


def _load_cached_rows(cache_path: Path) -> List[dict] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return data


def _write_cached_rows(cache_path: Path, rows: List[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)


def _epoch_to_utc_datetime(epoch: int) -> datetime:
    """Convert epoch seconds to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _git_last_commit_epoch(repo_path: Path, rel_path: str) -> int | None:
    """
    Return epoch seconds of the latest commit touching rel_path, or None.

    Uses plain git log (no --follow) and is intended for fallback checks only.
    """
    normalized = _normalize_path(rel_path)
    if not normalized:
        return None

    cmd = ["git", "-C", str(repo_path), "log", "-n", "1", "--format=%ct", "--", normalized]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        return int(lines[0])
    except ValueError:
        return None


def _git_first_commit_epoch_follow(repo_path: Path, rel_path: str) -> int | None:
    """
    Return epoch seconds of the earliest commit touching rel_path, or None.

    Uses --follow to recover history across renames for unresolved files.
    """
    normalized = _normalize_path(rel_path)
    if not normalized:
        return None

    cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--follow",
        "--reverse",
        "-n",
        "1",
        "--format=%ct",
        "--",
        normalized,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        return int(lines[0])
    except ValueError:
        return None


def _git_is_tracked(repo_path: Path, rel_path: str) -> bool:
    """Return True if rel_path is tracked by git in repo_path."""
    normalized = _normalize_path(rel_path)
    if not normalized:
        return False
    cmd = ["git", "-C", str(repo_path), "ls-files", "--error-unmatch", "--", normalized]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return False
    return proc.returncode == 0


def find_representative_files(repo_path: Path) -> Dict[str, int]:
    """Return mapping of relative file path -> LOC for representative files."""
    reps: Dict[str, int] = {}
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(repo_path).as_posix()
        if is_excluded_path(rel_path) or not is_included_path(rel_path):
            continue
        loc = _count_loc(file_path)
        reps[rel_path] = loc
    return reps


def _load_sonar_file_index(sonar_metrics_path: Path) -> Dict[str, set[str]]:
    """Load repo -> set(file_path) from sonar_metrics.csv, if available."""
    if not sonar_metrics_path.exists():
        print(f"[mine] Sonar metrics not found at {sonar_metrics_path}; using representative files.")
        return {}

    index: Dict[str, set[str]] = defaultdict(set)
    valid_repos = {config.repo_dir_name(url).lower() for url in config.ALL_REPOSITORIES}
    skipped_rows = 0
    try:
        with sonar_metrics_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "repo" not in reader.fieldnames or "file_path" not in reader.fieldnames:
                print(f"[mine] Sonar metrics missing repo/file_path; using representative files.")
                return {}
            for row in reader:
                repo_name = str(row.get("repo", "")).strip().lower()
                if repo_name not in valid_repos:
                    sonar_key = str(row.get("sonar_project_key", "")).strip().lower()
                    if sonar_key in valid_repos:
                        repo_name = sonar_key
                    else:
                        project_key = str(row.get("project_key", "")).strip().lower()
                        if project_key in valid_repos:
                            repo_name = project_key
                rel_path = str(row.get("file_path", "")).strip()
                if not repo_name or not rel_path:
                    continue
                if repo_name not in valid_repos:
                    skipped_rows += 1
                    continue
                rel_path = rel_path.replace("\\", "/")
                index[repo_name].add(rel_path)
    except Exception as exc:
        print(f"[mine] Failed to read {sonar_metrics_path}: {exc}; using representative files.")
        return {}
    total_files = sum(len(paths) for paths in index.values())
    print(f"[mine] Loaded Sonar file index for {len(index)} repos ({total_files} files)")
    if skipped_rows:
        print(f"[mine] Skipped {skipped_rows} Sonar rows that did not match known repos")
    return index


def _sonar_target_files(repo_path: Path, sonar_index: Dict[str, set[str]]) -> Dict[str, int]:
    """Return mapping of relative file path -> LOC for Sonar-identified files."""
    repo_key = repo_path.name.lower()
    sonar_paths = sonar_index.get(repo_key)
    if not sonar_paths:
        print(
            f"[mine] Sonar index has no entry for {repo_path.name} "
            f"(loaded repos={len(sonar_index)})"
        )
        return {}
    targets: Dict[str, int] = {}
    for raw_rel_path in sorted(sonar_paths):
        # Trust Sonar's file list for join coverage; only keep files that exist on disk.
        rel_path = raw_rel_path.replace("\\", "/").lstrip("/")
        file_path = repo_path / rel_path
        if not file_path.is_file():
            continue
        targets[rel_path] = _count_loc(file_path)
    return targets


def _resolve_target_files(
    repo_path: Path,
    sonar_paths: set[str] | None,
) -> Dict[str, int]:
    """Resolve target files from Sonar paths first, then representative fallback."""
    if sonar_paths:
        targets: Dict[str, int] = {}
        for raw_rel_path in sorted(sonar_paths):
            rel_path = raw_rel_path.replace("\\", "/").lstrip("/")
            file_path = repo_path / rel_path
            if not file_path.is_file():
                continue
            targets[rel_path] = _count_loc(file_path)
        if targets:
            print(f"[mine] Using Sonar file list for {repo_path.name} (n={len(targets)})")
            return targets
        print(
            f"[mine] Sonar list unavailable for {repo_path.name}; "
            "falling back to representative files"
        )

    return find_representative_files(repo_path)


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


def mine_repository(
    repo_path: Path,
    since: datetime,
    target_files: Dict[str, int] | None = None,
    repo_url_map: Dict[str, str] | None = None,
) -> List[dict]:
    """
    Traverse commits in the last WINDOW_DAYS to compute churn and knowledge-concentration risk.

    Churn: total added + removed lines per file.
    Single-contributor touch: file touched by exactly one author in the window.
    
    New fields:
    - recency_days: days since last commit touching the file
    - file_age_days: days since first commit touching the file (full history)
    - commit_count_12m: commits touching file in last 365 days
    - bus_factor_estimate: minimum k such that top-k authors >= 50% of file commits
    - git_metrics_status: "ok" if mining succeeded, "missing" if no git history found
    """
    analysis_date_utc = config.now_utc_iso()
    repo_head_sha = config.git_head_sha(repo_path)
    now_utc = datetime.now(timezone.utc)
    repo_start = time.perf_counter()
    if target_files is None:
        target_files = find_representative_files(repo_path)

    if not target_files:
        return []

    # Normalize all target file paths
    normalized_targets: Dict[str, str] = {}  # normalized -> original
    for rel_path in target_files:
        norm = _normalize_path(rel_path)
        normalized_targets[norm] = rel_path

    author_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    author_line_additions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    author_line_activity: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    churn: Dict[str, int] = defaultdict(int)
    added_totals: Dict[str, int] = defaultdict(int)
    deleted_totals: Dict[str, int] = defaultdict(int)
    path_aliases: Dict[str, str] = {}
    repo_commit_count = 0
    repo_authors: set[str] = set()
    author_display: Dict[str, str] = {}
    
    # Tracking for recency and file age
    file_last_commit_date: Dict[str, datetime] = {}  # Most recent commit per file
    file_commit_count: Dict[str, int] = defaultdict(int)  # Commits touching file in window
    files_touched_in_window: set[str] = set()  # Files that had any activity

    # Some pydriller versions do not support filepath_regex; filter manually instead.
    print(f"[mine]   12-month history pass...")
    t_12m_start = time.perf_counter()
    repo_mining = Repository(str(repo_path), since=since)
    for commit in repo_mining.traverse_commits():
        author = _author_key(commit.author)
        display = _author_display(commit.author)
        if author not in author_display or author_display[author] == "unknown":
            author_display[author] = display
        repo_commit_count += 1
        if repo_commit_count % 1000 == 0:
            print(f"[mine]   ...processed {repo_commit_count} commits (12m window)")
        repo_authors.add(author)
        
        commit_date = commit.committer_date
        if commit_date.tzinfo is None:
            commit_date = commit_date.replace(tzinfo=timezone.utc)

        mods = getattr(commit, "modifications", None)
        if mods is None:
            mods = getattr(commit, "modified_files", [])
        for mod in mods:
            new_path = mod.new_path
            old_path = mod.old_path
            if not new_path and not old_path:
                continue
            new_rel = _normalize_path(new_path) if new_path else ""
            old_rel = _normalize_path(old_path) if old_path else ""
            if old_rel and new_rel and old_rel != new_rel:
                canonical_new = _canonical_path(new_rel, path_aliases)
                path_aliases[old_rel] = canonical_new

            rel_path_norm = _canonical_path(new_rel or old_rel, path_aliases)
            if not rel_path_norm or rel_path_norm not in normalized_targets:
                continue
            
            # Use original path for storage
            rel_path = normalized_targets[rel_path_norm]
            files_touched_in_window.add(rel_path)

            added, deleted = _line_changes(mod)
            churn[rel_path] += added + deleted
            added_totals[rel_path] += added
            deleted_totals[rel_path] += deleted
            author_counts[rel_path][author] += 1
            author_line_additions[rel_path][author] += added
            author_line_activity[rel_path][author] += added + deleted
            
            # Track commit count and dates
            file_commit_count[rel_path] += 1
            
            # Update last commit date (most recent)
            if rel_path not in file_last_commit_date or commit_date > file_last_commit_date[rel_path]:
                file_last_commit_date[rel_path] = commit_date
            
    t_12m_s = time.perf_counter() - t_12m_start
    print(
        f"[mine]   12-month pass complete: {repo_commit_count} commits, "
        f"{len(files_touched_in_window)} files touched ({t_12m_s:.2f}s)"
    )

    # For file_age_days, we need full history - mine again without since filter
    # This is expensive, so we traverse in REVERSE chronological order (oldest first)
    # and can stop once we've found creation dates for all target files
    file_creation_date: Dict[str, datetime] = {}
    files_needing_creation_date = set(target_files.keys()) - files_touched_in_window
    t_full_history_s = 0.0
    if len(files_touched_in_window) == len(target_files):
        print("[mine]   Full history pass skipped (all targets touched in 12m)")
    elif files_needing_creation_date:
        print(
            f"[mine]   Full history pass for file creation dates "
            f"(tracking {len(files_needing_creation_date)} files)..."
        )
        t_full_start = time.perf_counter()
        try:
            # order='reverse' gives oldest commits first - efficient for creation dates
            full_repo_mining = Repository(str(repo_path), order='reverse')
            commit_count_full = 0
            for commit in full_repo_mining.traverse_commits():
                commit_count_full += 1
                if commit_count_full % 2000 == 0:
                    print(
                        f"[mine]     ...processed {commit_count_full} commits, "
                        f"found {len(file_creation_date)}/{len(target_files)} creation dates"
                    )

                # Early stop if we've found all pending files
                if not files_needing_creation_date:
                    print(f"[mine]   Found all creation dates after {commit_count_full} commits")
                    break

                commit_date = commit.committer_date
                if commit_date.tzinfo is None:
                    commit_date = commit_date.replace(tzinfo=timezone.utc)

                mods = getattr(commit, "modifications", None)
                if mods is None:
                    mods = getattr(commit, "modified_files", [])
                for mod in mods:
                    new_path = mod.new_path
                    old_path = mod.old_path
                    if not new_path and not old_path:
                        continue
                    new_rel = _normalize_path(new_path) if new_path else ""
                    old_rel = _normalize_path(old_path) if old_path else ""
                    if old_rel and new_rel and old_rel != new_rel:
                        canonical_new = _canonical_path(new_rel, path_aliases)
                        path_aliases[old_rel] = canonical_new

                    rel_path_norm = _canonical_path(new_rel or old_rel, path_aliases)
                    if not rel_path_norm or rel_path_norm not in normalized_targets:
                        continue

                    rel_path = normalized_targets[rel_path_norm]
                    if rel_path in files_needing_creation_date:
                        file_creation_date[rel_path] = commit_date
                        files_needing_creation_date.discard(rel_path)

            t_full_history_s = time.perf_counter() - t_full_start
            print(
                f"[mine]   Full history pass complete: {commit_count_full} commits, "
                f"{len(file_creation_date)} creation dates found ({t_full_history_s:.2f}s)"
            )
        except Exception as e:
            t_full_history_s = time.perf_counter() - t_full_start
            print(f"[mine] Warning: Failed to get full history for {repo_path.name}: {e}")

    # Fallback for unresolved files: query git directly (bounded to unresolved set only).
    unresolved_files = [
        rel_path
        for rel_path in target_files
        if rel_path not in files_touched_in_window and rel_path not in file_creation_date
    ]
    fallback_checked = len(unresolved_files)
    fallback_recovered = 0
    t_fallback_start = time.perf_counter()
    if fallback_checked == 0:
        print("[mine]   Fallback history check skipped (no unresolved files)")
    else:
        for rel_path in unresolved_files:
            # Cheap guard: if not tracked, skip expensive history calls.
            if not _git_is_tracked(repo_path, rel_path):
                continue

            latest_epoch = _git_last_commit_epoch(repo_path, rel_path)
            earliest_epoch = _git_first_commit_epoch_follow(repo_path, rel_path)
            recovered = False

            if latest_epoch is not None:
                file_last_commit_date[rel_path] = _epoch_to_utc_datetime(latest_epoch)
                recovered = True

            if earliest_epoch is not None:
                file_creation_date[rel_path] = _epoch_to_utc_datetime(earliest_epoch)
                recovered = True
            elif latest_epoch is not None:
                # Conservative fallback when only latest commit is available.
                file_creation_date[rel_path] = file_last_commit_date[rel_path]
                recovered = True

            if recovered:
                fallback_recovered += 1

        print(
            f"[mine]   Fallback history check: recovered "
            f"{fallback_recovered}/{fallback_checked} unresolved files"
        )
    t_fallback_s = time.perf_counter() - t_fallback_start
    print(f"[mine]   Fallback phase time: {t_fallback_s:.2f}s")

    repo_url_lookup = repo_url_map if repo_url_map is not None else _repo_url_lookup()
    repo_url = repo_url_lookup.get(repo_path.name, "")
    activity_label = (
        "ACTIVE"
        if repo_commit_count >= 24 and len(repo_authors) >= 3
        else "STAGNANT"
    )
    repo_truck_factor = _approx_truck_factor(author_counts)

    rows: List[dict] = []
    for rel_path, loc in target_files.items():
        authors_for_file = author_counts.get(rel_path, {})
        unique_authors = len(authors_for_file)
        total_modifications = sum(authors_for_file.values())
        
        # Determine if we have git history for this file
        # File exists in git if it was touched in 12m window, found in full history,
        # or recovered via fallback latest-commit lookup.
        has_git_history = (
            rel_path in files_touched_in_window
            or rel_path in file_creation_date
            or rel_path in file_last_commit_date
        )
        git_metrics_status = "ok" if has_git_history else "missing"
        
        # Compute recency_days
        # Priority: use last commit date in 12m window, else use creation date as "last known activity"
        recency_days = None
        if rel_path in file_last_commit_date:
            delta = now_utc - file_last_commit_date[rel_path]
            recency_days = max(0, delta.days)
        elif rel_path in file_creation_date:
            # File exists but no activity in 12m - use creation date (will be > 365 days)
            delta = now_utc - file_creation_date[rel_path]
            recency_days = max(0, delta.days)
        
        # Compute file_age_days (None if no history found)
        file_age_days = None
        if rel_path in file_creation_date:
            delta = now_utc - file_creation_date[rel_path]
            file_age_days = max(0, delta.days)
        
        # Commit count in 12m window (0 if file exists but no recent activity)
        commit_count_12m = file_commit_count.get(rel_path, 0) if has_git_history else None
        
        # Bus factor estimate for this file
        # If file exists in git but has no recent authors, bus_factor = 0 (no active contributors)
        bus_factor_estimate = _compute_file_bus_factor(authors_for_file)
        if has_git_history and bus_factor_estimate is None:
            bus_factor_estimate = 0  # File exists but no recent contributors
        
        # If no git history, set numeric fields to None (not 0)
        if git_metrics_status == "missing":
            churn_val = None
            added_val = None
            deleted_val = None
            unique_authors_val = None
            single_contributor_12m = None
            dominant_author = None
            dominant_author_share = None
            knowledge_concentration_flag_75 = None
        else:
            churn_val = churn.get(rel_path, 0)
            added_val = added_totals.get(rel_path, 0)
            deleted_val = deleted_totals.get(rel_path, 0)
            unique_authors_val = unique_authors
            single_contributor_12m = unique_authors == 1 and total_modifications > 0

            line_contribs = author_line_additions.get(rel_path, {})
            line_activity = author_line_activity.get(rel_path, {})
            total_added_lines = sum(line_contribs.values())
            total_activity_lines = sum(line_activity.values())

            dominant_author = ""
            dominant_author_share = 0.0
            dominant_author_key = ""

            if total_activity_lines > 0:
                dominant_author_key, max_lines = max(line_activity.items(), key=lambda kv: kv[1])
                dominant_author_share = max_lines / total_activity_lines if total_activity_lines else 0.0
            elif total_added_lines > 0:
                dominant_author_key, max_lines = max(line_contribs.items(), key=lambda kv: kv[1])
                dominant_author_share = max_lines / total_added_lines if total_added_lines else 0.0
            elif total_modifications > 0:
                dominant_author_key, max_mods = max(authors_for_file.items(), key=lambda kv: kv[1])
                dominant_author_share = max_mods / total_modifications

            if dominant_author_key:
                dominant_author = author_display.get(dominant_author_key, dominant_author_key)

            # Dominant author accounts for >= 75% of observed activity.
            knowledge_concentration_flag_75 = dominant_author_share >= 0.75 and (
                total_activity_lines > 0 or total_added_lines > 0 or total_modifications > 0
            )

        rows.append(
            {
                "repo": repo_path.name,
                "repo_url": repo_url,
                "activity_label": activity_label,
                "analysis_date_utc": analysis_date_utc,
                "repo_head_sha": repo_head_sha,
                "file_path": rel_path,
                "file_language": _infer_language(rel_path),
                "absolute_path": str((repo_path / rel_path).resolve()),
                "lines_of_code": loc,
                "git_metrics_status": git_metrics_status,
                "unique_authors_12m": unique_authors_val,
                "repo_unique_authors_12m": len(repo_authors),
                "repo_commits_12m": repo_commit_count,
                "repo_commits_per_month_12m": repo_commit_count / 12.0,
                "repo_approx_truck_factor_50": repo_truck_factor,
                "churn_12m": churn_val,
                "added_lines_12m": added_val,
                "deleted_lines_12m": deleted_val,
                "single_contributor_12m": single_contributor_12m,
                "knowledge_concentration_flag_75": knowledge_concentration_flag_75,
                "dominant_author": dominant_author,
                "dominant_author_share": dominant_author_share,
                "last_12m_observed_commits": total_modifications if git_metrics_status == "ok" else None,
                # New fields
                "recency_days": recency_days,
                "file_age_days": file_age_days,
                "commit_count_12m": commit_count_12m,
                "bus_factor_estimate": bus_factor_estimate,
            }
        )

    total_repo_s = time.perf_counter() - repo_start
    print(
        f"[mine]   Repo timings summary for {repo_path.name}: "
        f"12m={t_12m_s:.2f}s, full={t_full_history_s:.2f}s, "
        f"fallback={t_fallback_s:.2f}s, total={total_repo_s:.2f}s"
    )
    return rows


def _mine_single_repository(
    repo_url: str,
    since: datetime,
    sonar_paths: set[str] | None,
    use_cache: bool,
    refresh_cache: bool,
) -> tuple[str, List[dict]]:
    """Mine a single repository with optional cache lookup/write."""
    repo_path = config.RAW_REPOS_DIR / config.repo_dir_name(repo_url)
    repo_name = repo_path.name
    if not repo_path.exists():
        print(f"[mine] Missing repository at {repo_path}; skipping metrics.")
        return repo_name, []

    print(f"[mine] Mining git history for {repo_name}")

    t_targets_start = time.perf_counter()
    target_files = _resolve_target_files(repo_path, sonar_paths)
    t_targets_s = time.perf_counter() - t_targets_start
    print(f"[mine]   Target resolution: {len(target_files)} files ({t_targets_s:.2f}s)")
    if not target_files:
        return repo_name, []

    since_date = since.date().isoformat()
    repo_head_sha = config.git_head_sha(repo_path) or "unknown"
    cache_path = _miner_cache_path(repo_name, repo_head_sha, since_date, target_files)
    if use_cache and not refresh_cache:
        cached_rows = _load_cached_rows(cache_path)
        if cached_rows is not None:
            print(f"[mine]   Cache hit: {cache_path.name} ({len(cached_rows)} rows)")
            return repo_name, cached_rows

    rows = mine_repository(
        repo_path=repo_path,
        since=since,
        target_files=target_files,
        repo_url_map={repo_name: repo_url},
    )

    if use_cache:
        try:
            _write_cached_rows(cache_path, rows)
            print(f"[mine]   Cache write: {cache_path.name}")
        except OSError as exc:
            print(f"[mine]   Cache write failed for {repo_name}: {exc}")

    return repo_name, rows


def run_git_mining(
    output_path: Path | None = None,
    sonar_metrics_path: Path | None = None,
    repo_urls: Iterable[str] | None = None,
    workers: int | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Path:
    """Mine all configured repositories and persist git metrics CSV."""
    output = output_path or config.git_metrics_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    config.ensure_data_dirs()
    _miner_cache_dir().mkdir(parents=True, exist_ok=True)

    since = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    all_rows: List[dict] = []
    sonar_path = sonar_metrics_path or config.sonar_metrics_path()
    sonar_file_index = _load_sonar_file_index(sonar_path)
    selected_repo_urls = list(repo_urls) if repo_urls is not None else list(config.ALL_REPOSITORIES)
    if not selected_repo_urls:
        print("[mine] No repositories selected.")
        selected_repo_urls = []

    max_workers = workers if workers is not None else min(4, (os.cpu_count() or 1))
    max_workers = max(1, int(max_workers))
    if refresh_cache and not use_cache:
        refresh_cache = False

    print(
        f"[mine] Run settings: repos={len(selected_repo_urls)}, workers={max_workers}, "
        f"use_cache={use_cache}, refresh_cache={refresh_cache}"
    )

    tasks: list[tuple[str, datetime, set[str] | None, bool, bool]] = []
    for repo_url in selected_repo_urls:
        repo_name = config.repo_dir_name(repo_url)
        sonar_paths = sonar_file_index.get(repo_name.lower())
        tasks.append((repo_url, since, sonar_paths, use_cache, refresh_cache))

    if max_workers == 1 or len(tasks) <= 1:
        for task in tasks:
            _, rows = _mine_single_repository(*task)
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_mine_single_repository, *task): task[0] for task in tasks}
            for future in as_completed(future_map):
                repo_url = future_map[future]
                try:
                    _, rows = future.result()
                    all_rows.extend(rows)
                except Exception as exc:
                    repo_name = config.repo_dir_name(repo_url)
                    print(f"[mine] Failed mining for {repo_name}: {exc}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.sort_values(["repo", "file_path"], inplace=True)

    archived = config.archive_existing_csv(output)
    if archived:
        print(f"[mine] Archived previous git metrics to {archived}")

    df.to_csv(output, index=False)
    print(f"[mine] Wrote git metrics to {output}")
    return output


def _parse_repositories_arg(repos_arg: str | None) -> list[str]:
    """
    Parse --repos argument as comma-separated repo directory names or URLs.
    """
    if not repos_arg:
        return list(config.ALL_REPOSITORIES)
    tokens = [token.strip() for token in repos_arg.split(",") if token.strip()]
    if not tokens:
        return list(config.ALL_REPOSITORIES)

    all_by_name = {config.repo_dir_name(url): url for url in config.ALL_REPOSITORIES}
    selected: list[str] = []
    for token in tokens:
        if token.startswith("http://") or token.startswith("https://"):
            selected.append(token)
            continue
        repo_url = all_by_name.get(token)
        if repo_url:
            selected.append(repo_url)
        else:
            print(f"[mine] Unknown repo token '{token}', skipping.")
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run git mining metrics.")
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated repo names (e.g. axios,nest) or full repo URLs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(4, CPU count)).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing per-repo mining cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute and overwrite cache entries for selected repos.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    selected_repo_urls = _parse_repositories_arg(args.repos)
    clone_repositories(selected_repo_urls)
    run_git_mining(
        repo_urls=selected_repo_urls,
        workers=args.workers,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
