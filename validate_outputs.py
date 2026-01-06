"""
Validate pipeline outputs for consistency.

Checks required files/columns and join coverage between git, sonar, and LLM metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from configs import config

RESULTS_DIR = config.RESULTS_DIR
REPORT_PATH = RESULTS_DIR / "validate_report.json"

FILES = {
    "git_metrics.csv": RESULTS_DIR / "git_metrics.csv",
    "sonar_metrics.csv": RESULTS_DIR / "sonar_metrics.csv",
    "llm_metrics.csv": RESULTS_DIR / "llm_metrics.csv",
}

REQUIRED_COLUMNS = {
    "git_metrics.csv": {"repo", "file_path", "absolute_path"},
    "llm_metrics.csv": {"repo", "file_path", "readability_score", "maintainability_risk"},
}
SONAR_KEY_COLUMNS = {"project_key", "sonar_project_key"}


def _load_df(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def _missing_columns_for_file(name: str, df: pd.DataFrame | None) -> list[str]:
    if df is None:
        if name == "sonar_metrics.csv":
            return ["repo", "file_path", "project_key|sonar_project_key"]
        return sorted(REQUIRED_COLUMNS.get(name, set()))
    if df.empty:
        return []

    if name == "sonar_metrics.csv":
        missing = []
        for col in ("repo", "file_path"):
            if col not in df.columns:
                missing.append(col)
        if not (SONAR_KEY_COLUMNS & set(df.columns)):
            missing.append("project_key|sonar_project_key")
        return missing

    required = REQUIRED_COLUMNS.get(name, set())
    return sorted(set(required) - set(df.columns))


def _join_coverage(base_df: pd.DataFrame | None, other_df: pd.DataFrame | None) -> dict | None:
    if base_df is None or other_df is None:
        return None
    if base_df.empty or other_df.empty:
        return None
    if not {"repo", "file_path"}.issubset(base_df.columns):
        return None
    if not {"repo", "file_path"}.issubset(other_df.columns):
        return None

    base_keys = list(zip(base_df["repo"], base_df["file_path"]))
    other_keys = set(zip(other_df["repo"], other_df["file_path"]))
    if not base_keys:
        return None
    matched = sum(1 for key in base_keys if key in other_keys)
    base_rows = len(base_keys)
    match_pct = (matched / base_rows) * 100 if base_rows else None
    return {
        "matched_rows": matched,
        "base_rows": base_rows,
        "match_pct": match_pct,
    }


def _duplicate_stats(df: pd.DataFrame | None) -> dict | None:
    if df is None or df.empty:
        return None
    if not {"repo", "file_path"}.issubset(df.columns):
        return None
    dup_mask = df.duplicated(subset=["repo", "file_path"], keep=False)
    duplicate_rows = int(dup_mask.sum())
    if duplicate_rows == 0:
        return {"duplicate_rows": 0, "duplicate_keys": 0}
    duplicate_keys = int(df.loc[dup_mask, ["repo", "file_path"]].drop_duplicates().shape[0])
    return {"duplicate_rows": duplicate_rows, "duplicate_keys": duplicate_keys}


def _unmatched_examples(
    base_df: pd.DataFrame | None, other_df: pd.DataFrame | None, limit: int = 20
) -> list[str]:
    if base_df is None or other_df is None:
        return []
    if base_df.empty or other_df.empty:
        return []
    if not {"repo", "file_path"}.issubset(base_df.columns):
        return []
    if not {"repo", "file_path"}.issubset(other_df.columns):
        return []
    try:
        base_keys = base_df[["repo", "file_path"]].copy()
        other_keys = other_df[["repo", "file_path"]].drop_duplicates()
        merged = base_keys.merge(other_keys, on=["repo", "file_path"], how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"].head(limit)
        return [f"{row.repo}/{row.file_path}" for row in missing.itertuples()]
    except Exception:
        return []


def run_validation(output_path: Path | None = None) -> Path:
    config.ensure_data_dirs()
    report_path = output_path or REPORT_PATH

    report = {
        "generated_at_utc": config.now_utc_iso(),
        "files": {},
        "join_coverage": {},
        "duplicates": {},
        "unmatched_examples": {},
        "error": None,
    }

    try:
        dataframes: dict[str, pd.DataFrame | None] = {}
        file_reports: dict[str, dict] = {}

        for name, path in FILES.items():
            df = _load_df(path)
            dataframes[name] = df
            missing_cols = _missing_columns_for_file(name, df)
            rows = 0 if df is None else len(df)
            file_reports[name] = {
                "path": str(path),
                "exists": path.exists(),
                "rows": rows,
                "missing_columns": missing_cols,
            }

        git_df = dataframes["git_metrics.csv"]
        sonar_df = dataframes["sonar_metrics.csv"]
        llm_df = dataframes["llm_metrics.csv"]

        coverage_git_sonar = _join_coverage(git_df, sonar_df)
        coverage_git_llm = _join_coverage(git_df, llm_df)

        report["files"] = file_reports
        report["join_coverage"] = {
            "git_to_sonar": coverage_git_sonar,
            "git_to_llm": coverage_git_llm,
        }
        report["duplicates"] = {
            "git_metrics.csv": _duplicate_stats(git_df),
            "sonar_metrics.csv": _duplicate_stats(sonar_df),
            "llm_metrics.csv": _duplicate_stats(llm_df),
        }
        report["unmatched_examples"] = {
            "git_missing_in_sonar": _unmatched_examples(git_df, sonar_df),
            "git_missing_in_llm": _unmatched_examples(git_df, llm_df),
        }

        print("[validate] Output validation report")
        for name, info in file_reports.items():
            status = "OK" if info["exists"] else "MISSING"
            missing = info["missing_columns"]
            missing_note = f" missing_columns={missing}" if missing else ""
            print(f"[validate] {name}: {status} rows={info['rows']}{missing_note}")

        def _format_cov(label: str, cov: dict | None) -> None:
            if not cov:
                print(f"[validate] {label}: n/a")
                return
            pct = cov["match_pct"]
            pct_str = f"{pct:.2f}%" if pct is not None else "n/a"
            print(
                f"[validate] {label}: {pct_str} (matched={cov['matched_rows']}/{cov['base_rows']})"
            )

        _format_cov("git->sonar", coverage_git_sonar)
        _format_cov("git->llm", coverage_git_llm)

        for name, stats in report["duplicates"].items():
            if not stats:
                print(f"[validate] duplicates {name}: n/a")
                continue
            print(
                f"[validate] duplicates {name}: rows={stats['duplicate_rows']} keys={stats['duplicate_keys']}"
            )
        print(
            f"[validate] unmatched git->sonar: {len(report['unmatched_examples']['git_missing_in_sonar'])}"
        )
        print(
            f"[validate] unmatched git->llm: {len(report['unmatched_examples']['git_missing_in_llm'])}"
        )
    except Exception as exc:
        report["error"] = str(exc)
        print(f"[validate] ERROR: {exc}")

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[validate] Wrote {report_path}")
    return report_path


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()
