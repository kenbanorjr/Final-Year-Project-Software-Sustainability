"""
Repository-level summary derived from file-level metrics.

Aggregates git, sonar, and LLM outputs into llm_repo_summary.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.configs import config

RESULTS_DIR = config.RESULTS_DIR
LLM_METRICS_PATH = config.llm_metrics_path(prefer_existing=True)
GIT_METRICS_PATH = config.git_metrics_path()
SONAR_METRICS_PATH = config.sonar_metrics_path()
OUTPUT_PATH = config.llm_repo_summary_path()


def _load_metrics(path: Path, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    if required:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    return df


def _normalize_llm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "llm_success" in df.columns:
        df = df[df["llm_success"].astype(str).str.lower().isin({"true", "1", "yes"})]
    df["readability_score"] = pd.to_numeric(df.get("readability_score"), errors="coerce")
    if "maintainability_risk" in df.columns:
        df["maintainability_risk"] = (
            df["maintainability_risk"].astype(str).str.strip().str.lower()
        )
    else:
        df["maintainability_risk"] = ""
    if "repo" not in df.columns:
        df["repo"] = "UNKNOWN"
    df["repo"] = df["repo"].fillna("UNKNOWN")
    return df


def _normalize_git_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ("churn_12m", "dominant_author_share", "unique_authors_12m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "single_contributor_12m" not in df.columns:
        if "single_author_touch_12m" in df.columns:
            df["single_contributor_12m"] = df["single_author_touch_12m"]
        elif "bus_factor_single_dev" in df.columns:
            df["single_contributor_12m"] = df["bus_factor_single_dev"]
    if "single_contributor_12m" in df.columns:
        df["single_contributor_12m"] = df["single_contributor_12m"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    if "repo" not in df.columns:
        df["repo"] = "UNKNOWN"
    df["repo"] = df["repo"].fillna("UNKNOWN")
    return df


def _normalize_sonar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    numeric_cols = [
        "sonar_complexity",
        "sonar_sqale_index",
        "sonar_code_smells",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "repo" not in df.columns:
        df["repo"] = df.get("project_key")
    df["repo"] = df["repo"].fillna("UNKNOWN")
    return df


def _mean_value(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    value = df[column].mean()
    if pd.isna(value):
        return None
    return float(value)


def _most_common(values: pd.Series | None) -> str | None:
    if values is None:
        return None
    series = values.dropna()
    if series.empty:
        return None
    return series.value_counts().idxmax()


def _aggregate_repo(repo: str, llm_df: pd.DataFrame, git_df: pd.DataFrame, sonar_df: pd.DataFrame) -> dict:
    repo_llm = llm_df[llm_df["repo"] == repo]
    repo_git = git_df[git_df["repo"] == repo]
    repo_sonar = sonar_df[sonar_df["repo"] == repo]

    llm_files = len(repo_llm)
    high_count = 0
    if llm_files and "maintainability_risk" in repo_llm.columns:
        high_count = (repo_llm["maintainability_risk"] == "high").sum()
    maintainability_risk_high_pct = (
        (high_count / llm_files) * 100.0 if llm_files else None
    )

    single_contributor_rate = None
    if not repo_git.empty and "single_contributor_12m" in repo_git.columns:
        single_contributor_rate = repo_git["single_contributor_12m"].astype(float).mean()
        if pd.isna(single_contributor_rate):
            single_contributor_rate = None

    return {
        "repo": repo,
        "mean_readability_score": _mean_value(repo_llm, "readability_score"),
        "maintainability_risk_high_pct": maintainability_risk_high_pct,
        "mean_sonar_sqale_index": _mean_value(repo_sonar, "sonar_sqale_index"),
        "mean_sonar_code_smells": _mean_value(repo_sonar, "sonar_code_smells"),
        "mean_sonar_complexity": _mean_value(repo_sonar, "sonar_complexity"),
        "mean_churn_12m": _mean_value(repo_git, "churn_12m"),
        "mean_unique_authors_12m": _mean_value(repo_git, "unique_authors_12m"),
        "dominant_author_share_mean": _mean_value(repo_git, "dominant_author_share"),
        "single_contributor_12m_rate": single_contributor_rate,
        "activity_label": _most_common(repo_git.get("activity_label")),
    }


def _compute_thresholds(summary_df: pd.DataFrame) -> dict[str, float | None]:
    thresholds: dict[str, float | None] = {}
    for col, key in (
        ("mean_sonar_sqale_index", "sqale"),
        ("maintainability_risk_high_pct", "llm_risk"),
        ("mean_sonar_complexity", "complexity"),
    ):
        series = summary_df[col].dropna() if col in summary_df.columns else pd.Series(dtype=float)
        thresholds[key] = series.quantile(0.75) if not series.empty else None
    return thresholds


def _repo_summary(row: pd.Series, thresholds: dict[str, float | None]) -> str:
    def is_high(value: float | None, threshold: float | None) -> bool:
        return value is not None and threshold is not None and value >= threshold

    if all(
        row.get(col) is None
        for col in (
            "maintainability_risk_high_pct",
            "mean_sonar_sqale_index",
            "single_contributor_12m_rate",
        )
    ):
        return "Insufficient data"

    high_llm = is_high(row.get("maintainability_risk_high_pct"), thresholds.get("llm_risk"))
    high_debt = is_high(row.get("mean_sonar_sqale_index"), thresholds.get("sqale"))
    high_complexity = is_high(row.get("mean_sonar_complexity"), thresholds.get("complexity"))
    bus_factor_rate = row.get("single_contributor_12m_rate") or 0.0
    dominant_share = row.get("dominant_author_share_mean") or 0.0
    high_bus_factor = bus_factor_rate >= 0.5 or dominant_share >= 0.75

    if high_debt and high_llm:
        return "High technical risk"
    if high_bus_factor:
        return "High knowledge concentration risk"
    if high_llm and high_complexity:
        return "Elevated maintainability risk"
    if high_debt:
        return "Elevated technical debt risk"
    return "Moderate risk profile"


def run_repo_summaries(output_path: Path | None = None) -> Path:
    config.ensure_data_dirs()
    output = output_path or OUTPUT_PATH

    llm_df = _normalize_llm_df(_load_metrics(LLM_METRICS_PATH, ["repo", "file_path"]))
    git_df = _normalize_git_df(_load_metrics(GIT_METRICS_PATH))
    sonar_df = _normalize_sonar_df(_load_metrics(SONAR_METRICS_PATH))

    if llm_df.empty and git_df.empty and sonar_df.empty:
        raise RuntimeError("No metrics found; run miner.py, sonar_runner.py, and llm_judge.py first.")

    repos = sorted(
        set(llm_df.get("repo", pd.Series(dtype=str)).dropna().tolist())
        | set(git_df.get("repo", pd.Series(dtype=str)).dropna().tolist())
        | set(sonar_df.get("repo", pd.Series(dtype=str)).dropna().tolist())
    )

    rows = [_aggregate_repo(repo, llm_df, git_df, sonar_df) for repo in repos]
    summary_df = pd.DataFrame(rows)
    thresholds = _compute_thresholds(summary_df)
    summary_df["repo_summary"] = summary_df.apply(
        lambda row: _repo_summary(row, thresholds), axis=1
    )
    summary_df.to_csv(output, index=False)
    print(f"[llm-repo] Wrote repo summaries to {output}")
    return output


def main() -> None:
    run_repo_summaries()


if __name__ == "__main__":
    main()
