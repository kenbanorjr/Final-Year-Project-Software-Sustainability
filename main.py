"""
Orchestrator for the sustainability evaluation pipeline.

Steps:
1. Clone repositories.
2. Mine git history for churn/bus-factor on representative files.
3. Run SonarQube scans and collect baseline metrics.
4. Run the LLM judge on representative files.
5. Merge outputs into final_dataset.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from configs import config
import llm_judge
import llm_repo_summary
import miner
import sonar_runner
import validate_outputs


def _repo_paths() -> list[Path]:
    return [config.RAW_REPOS_DIR / config.repo_dir_name(url) for url in config.ALL_REPOSITORIES]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists and has content; otherwise return empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _drop_overlapping_columns(df: pd.DataFrame, existing_cols: set[str]) -> pd.DataFrame:
    """Drop non-key columns that would collide on merge."""
    if df.empty:
        return df
    protected = {"repo", "file_path"}
    overlap = (set(df.columns) & existing_cols) - protected
    if not overlap:
        return df
    return df.drop(columns=sorted(overlap))


def merge_outputs(
    git_metrics_csv: Path, sonar_metrics_csv: Path, llm_metrics_csv: Path, output_path: Path
) -> Path:
    """Join the three result sets on repo/file_path."""
    git_df = _safe_read_csv(git_metrics_csv)
    sonar_df = _safe_read_csv(sonar_metrics_csv)
    llm_df = _safe_read_csv(llm_metrics_csv)

    if "seed_category" not in git_df.columns and "category" in git_df.columns:
        git_df = git_df.rename(columns={"category": "seed_category"})

    merged = git_df
    if not sonar_df.empty:
        sonar_df = _drop_overlapping_columns(sonar_df, set(merged.columns))
        merged = merged.merge(sonar_df, on=["repo", "file_path"], how="left")
    if not llm_df.empty:
        llm_df = _drop_overlapping_columns(llm_df, set(merged.columns))
        merged = merged.merge(llm_df, on=["repo", "file_path"], how="left")

    merged.to_csv(output_path, index=False)
    print(f"[main] Wrote merged dataset to {output_path}")
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sustainability evaluation pipeline.")
    parser.add_argument(
        "--with-repo-summary",
        action="store_true",
        help="Generate llm_repo_summary.csv after merging datasets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config.ensure_data_dirs()

    print("[main] Step 1: clone repositories")
    miner.clone_repositories(config.ALL_REPOSITORIES)

    print("[main] Step 2: git mining")
    git_metrics_csv = miner.run_git_mining()

    print("[main] Step 3: SonarQube baseline")
    repo_paths = _repo_paths()
    sonar_metrics_csv = sonar_runner.collect_sonar_metrics(repo_paths)

    print("[main] Step 4: LLM judge")
    llm_metrics_csv = llm_judge.run_llm_judge(git_metrics_csv)

    print("[main] Step 5: merge datasets")
    merge_outputs(
        git_metrics_csv=git_metrics_csv,
        sonar_metrics_csv=sonar_metrics_csv,
        llm_metrics_csv=llm_metrics_csv,
        output_path=config.RESULTS_DIR / "final_dataset.csv",
    )
    if args.with_repo_summary:
        print("[main] Step 6: repo summary")
        llm_repo_summary.run_repo_summaries()
    if config.VALIDATE_OUTPUTS:
        print("[main] Step 7: validate outputs")
        validate_outputs.run_validation()


if __name__ == "__main__":
    main()
