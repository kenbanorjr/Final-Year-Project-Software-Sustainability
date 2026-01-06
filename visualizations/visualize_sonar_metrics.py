"""
Sonar Metrics Visualization Script
Generates tables and charts for analyzing SonarQube file-level metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/results/sonar_metrics.csv")
OUTPUT_DIR = Path("data/results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ISSUE_COLUMNS = [
    "sonar_bugs",
    "sonar_vulnerabilities",
    "sonar_code_smells",
    "sonar_violations",
]


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("sonar_metrics.csv is empty")

    numeric_cols = [
        "sonar_complexity",
        "sonar_ncloc",
        "sonar_cognitive_complexity",
        "sonar_comment_lines_density",
        "sonar_reliability_rating",
        "sonar_security_rating",
        "sonar_sqale_rating",
        "sonar_sqale_index",
        "sonar_duplicated_blocks",
        "sonar_duplicated_lines_density",
        "sonar_test_success_density",
    ] + ISSUE_COLUMNS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "repo" not in df.columns:
        df["repo"] = pd.NA
    fallback_repo = df["project_key"] if "project_key" in df.columns else pd.NA
    df["repo"] = df["repo"].fillna(fallback_repo).fillna("UNKNOWN")
    return df


def plot_repo_distributions(df: pd.DataFrame) -> None:
    metrics = [
        ("sonar_complexity", "Complexity"),
        ("sonar_cognitive_complexity", "Cognitive Complexity"),
        ("sonar_sqale_index", "SQALE Index"),
    ]
    repo_order = sorted(df["repo"].dropna().unique().tolist())

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Sonar Metrics Distribution by Repository", fontsize=14, fontweight="bold")

    for ax, (metric, label) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        data = df[["repo", metric]].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        sns.violinplot(
            data=data,
            x="repo",
            y=metric,
            order=repo_order,
            inner="quartile",
            cut=0,
            ax=ax,
            color="#5DA5DA",
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sonar_repo_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_issue_bars(df: pd.DataFrame) -> None:
    issue_cols = _available_columns(df, ISSUE_COLUMNS)
    if not issue_cols:
        print("Skipping issue bar chart: no issue columns in data.")
        return
    issue_df = df.groupby("repo")[issue_cols].sum().fillna(0)
    issue_df = issue_df.sort_values(by=issue_cols[0], ascending=False)

    ax = issue_df.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6),
        color=["#e74c3c", "#8e44ad", "#f39c12", "#3498db"],
    )
    ax.set_title("Issue Counts by Repository (Stacked)")
    ax.set_xlabel("Repository")
    ax.set_ylabel("Count")
    ax.legend([c.replace("sonar_", "").replace("_", " ").title() for c in issue_cols])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sonar_issue_counts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap(df: pd.DataFrame) -> None:
    agg_spec = {}
    if "sonar_ncloc" in df.columns:
        agg_spec["sonar_ncloc"] = ("sonar_ncloc", "sum")
    if "sonar_complexity" in df.columns:
        agg_spec["complexity_mean"] = ("sonar_complexity", "mean")
    if "sonar_cognitive_complexity" in df.columns:
        agg_spec["cognitive_mean"] = ("sonar_cognitive_complexity", "mean")
    if "sonar_sqale_index" in df.columns:
        agg_spec["sqale_mean"] = ("sonar_sqale_index", "mean")
    if "sonar_duplicated_lines_density" in df.columns:
        agg_spec["duplicated_density_mean"] = ("sonar_duplicated_lines_density", "mean")
    if "sonar_comment_lines_density" in df.columns:
        agg_spec["comment_density_mean"] = ("sonar_comment_lines_density", "mean")
    if "sonar_bugs" in df.columns:
        agg_spec["bugs_total"] = ("sonar_bugs", "sum")
    if "sonar_vulnerabilities" in df.columns:
        agg_spec["vulns_total"] = ("sonar_vulnerabilities", "sum")
    if "sonar_code_smells" in df.columns:
        agg_spec["code_smells_total"] = ("sonar_code_smells", "sum")
    if "sonar_violations" in df.columns:
        agg_spec["violations_total"] = ("sonar_violations", "sum")

    if not agg_spec:
        print("Skipping heatmap: no aggregateable metrics in data.")
        return

    repo_agg = df.groupby("repo").agg(**agg_spec)

    ncloc = repo_agg["sonar_ncloc"].replace(0, pd.NA) if "sonar_ncloc" in repo_agg.columns else None
    if ncloc is not None and "bugs_total" in repo_agg.columns:
        repo_agg["bugs_per_kloc"] = repo_agg["bugs_total"] / (ncloc / 1000)
    if ncloc is not None and "vulns_total" in repo_agg.columns:
        repo_agg["vulns_per_kloc"] = repo_agg["vulns_total"] / (ncloc / 1000)
    if ncloc is not None and "code_smells_total" in repo_agg.columns:
        repo_agg["smells_per_kloc"] = repo_agg["code_smells_total"] / (ncloc / 1000)
    if ncloc is not None and "violations_total" in repo_agg.columns:
        repo_agg["violations_per_kloc"] = repo_agg["violations_total"] / (ncloc / 1000)

    heatmap_cols = [
        "complexity_mean",
        "cognitive_mean",
        "sqale_mean",
        "duplicated_density_mean",
        "comment_density_mean",
        "bugs_per_kloc",
        "vulns_per_kloc",
        "smells_per_kloc",
        "violations_per_kloc",
    ]
    heatmap_cols = _available_columns(repo_agg, heatmap_cols)
    if not heatmap_cols:
        print("Skipping heatmap: no usable metrics after aggregation.")
        return
    heatmap_data = repo_agg[heatmap_cols].fillna(0)
    normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    normalized = normalized.fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Normalized Value (0-1)"},
    )
    plt.title("Normalized Sonar Metrics by Repository")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sonar_metrics_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter_matrix(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sonar Metrics Relationships", fontsize=14, fontweight="bold")

    plot_specs = [
        ("sonar_complexity", "sonar_ncloc", "Complexity vs NCLOC", "Complexity", "NCLOC", "#3498db"),
        ("sonar_sqale_index", "sonar_code_smells", "SQALE Index vs Code Smells", "SQALE Index", "Code Smells", "#e67e22"),
        (
            "sonar_duplicated_lines_density",
            "sonar_ncloc",
            "Duplication vs NCLOC",
            "Duplicated Lines Density",
            "NCLOC",
            "#8e44ad",
        ),
    ]

    for ax, (x_col, y_col, title, x_label, y_label, color) in zip(axes, plot_specs):
        if x_col not in df.columns or y_col not in df.columns:
            ax.set_visible(False)
            continue
        ax.scatter(df[x_col], df[y_col], alpha=0.4, s=10, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sonar_scatter_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_histograms(df: pd.DataFrame) -> None:
    metrics = [
        ("sonar_complexity", "Complexity"),
        ("sonar_cognitive_complexity", "Cognitive Complexity"),
        ("sonar_sqale_index", "SQALE Index"),
        ("sonar_duplicated_lines_density", "Duplicated Lines Density"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Distribution of Key Sonar Metrics", fontsize=14, fontweight="bold")

    for ax, (metric, label) in zip(axes.flatten(), metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        data = df[metric].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=30, color="#2ecc71", edgecolor="black", alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("File Count")
        ax.set_yscale("log")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sonar_metric_histograms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    print("Loading Sonar metrics data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows across {df['repo'].nunique()} repositories")

    plot_repo_distributions(df)
    plot_issue_bars(df)
    plot_heatmap(df)
    plot_scatter_matrix(df)
    plot_histograms(df)

    print(f"All outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
