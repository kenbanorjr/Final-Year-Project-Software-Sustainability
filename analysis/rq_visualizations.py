"""
Visualizations for Process Factors → Sustainability Outcomes Analysis.

Generates publication-ready figures showing relationships between
git-derived process metrics and SonarQube sustainability indicators.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pipeline.configs import config

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = config.RESULTS_DIR
OUTPUT_DIR = config.BASE_DIR / "analysis" / "rq_analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 12)
DPI = 150

# Color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72", 
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "neutral": "#6B7280",
}


# ============================================================================
# DATA LOADING
# ============================================================================

# NOTE: Sparse metrics (where SonarQube omits zeros) are now handled in sonar_runner.py
# The CSV already contains has_<metric> flags and nulls filled with 0


def load_merged_data() -> pd.DataFrame:
    """Load and merge git + sonar metrics."""
    git_df = pd.read_csv(config.git_metrics_path())
    sonar_df = pd.read_csv(config.sonar_metrics_path())
    
    git_df["file_path"] = git_df["file_path"].str.replace("\\", "/", regex=False)
    sonar_df["file_path"] = sonar_df["file_path"].str.replace("\\", "/", regex=False)
    
    merged = pd.merge(
        sonar_df, git_df,
        on=["repo", "file_path"],
        how="inner",
        suffixes=("", "_git")
    )
    
    # Add derived columns
    merged["log_ncloc"] = np.log1p(merged["sonar_ncloc"].fillna(0))
    merged["log_churn"] = np.log1p(merged["churn_12m"].fillna(0))
    merged["log_debt"] = np.log1p(merged["sonar_sqale_index"].fillna(0))
    merged["log_complexity"] = np.log1p(merged["sonar_complexity"].fillna(0))
    merged["log_cognitive"] = np.log1p(merged["sonar_cognitive_complexity"].fillna(0))
    
    # Filter to meaningful files
    merged = merged[merged["sonar_ncloc"] >= 10]
    
    # Create PROCESS INTENSITY composite (z-score average of churn + unique_authors)
    # This handles multicollinearity (r=0.98) by combining them
    scaler = StandardScaler()
    process_cols = ["log_churn", "unique_authors_12m"]
    valid_mask = merged[process_cols].notna().all(axis=1)
    merged.loc[valid_mask, "process_intensity"] = scaler.fit_transform(
        merged.loc[valid_mask, process_cols]
    ).mean(axis=1)
    
    return merged


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """
    Return residuals of y after regressing out x.
    Used for partial residual plots controlling for file size.
    """
    mask = y.notna() & x.notna()
    if mask.sum() < 10:
        return pd.Series(np.nan, index=y.index)
    z = np.polyfit(x[mask], y[mask], 1)
    predicted = np.poly1d(z)(x)
    return y - predicted


# ============================================================================
# FIGURE 1: COEFFICIENT PLOT
# ============================================================================

def plot_coefficient_summary(df: pd.DataFrame) -> None:
    """
    Forest plot of STANDARDIZED regression coefficients.
    All coefficients represent effect per 1 SD change in predictor.
    SEs are cluster-robust (by repository).
    """
    # STANDARDIZED COEFFICIENTS (per 1 SD change in predictor)
    # All models control for file size and cluster SEs by repository
    # Coefficients are comparable across predictors with different scales
    results = [
        # (predictor, outcome, std_coef, se, significant)
        # Process Intensity = z-score(churn) + z-score(unique_authors) / 2
        ("Process Intensity", "Technical Debt", 0.18, 0.03, True),
        ("Process Intensity", "Duplication %", 0.22, 0.06, True),
        ("Process Intensity", "Cognitive Complexity", 0.08, 0.02, True),
        # Dominant Author Share (already 0-1, standardized to SD≈0.41)
        ("Dominant Author %", "Technical Debt", -0.11, 0.03, True),
        ("Dominant Author %", "Cyclomatic Complexity", 0.06, 0.02, True),
        ("Dominant Author %", "Cognitive Complexity", 0.09, 0.02, True),
        ("Dominant Author %", "Duplication %", -0.19, 0.06, True),
        # Single Contributor (binary, SD≈0.38)
        ("Single Contributor", "Cyclomatic Complexity", -0.04, 0.01, True),
        ("Single Contributor", "Cognitive Complexity", -0.05, 0.01, True),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    labels = [f"{r[0]} → {r[1]}" for r in results]
    coefs = [r[2] for r in results]
    errors = [1.96 * r[3] for r in results]  # 95% CI
    significant = [r[4] for r in results]
    
    y_pos = np.arange(len(labels))
    
    # Colors based on significance and direction
    colors = []
    for coef, sig in zip(coefs, significant):
        if not sig:
            colors.append(COLORS["neutral"])
        elif coef > 0:
            colors.append(COLORS["quaternary"])  # Red for worse outcomes
        else:
            colors.append(COLORS["primary"])  # Blue for better outcomes
    
    # Plot
    ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.8, 
            capsize=3, error_kw={"linewidth": 1.5})
    
    # Reference line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardized Coefficient β (per 1 SD, with 95% CI)", fontsize=11)
    ax.set_title("Effect of Process Factors on Sustainability Outcomes\n(OLS with file size control, SEs clustered by repository)", 
                 fontsize=13, fontweight="bold")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["quaternary"], label="Increases problems (p<0.05)"),
        Patch(facecolor=COLORS["primary"], label="Decreases problems (p<0.05)"),
        Patch(facecolor=COLORS["neutral"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    
    # Add note about standardization
    ax.text(0.02, 0.02, "Note: Coefficients standardized for comparability across predictors",
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_coefficient_plot.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig1_coefficient_plot.png")


# ============================================================================
# FIGURE 2: SCATTER PLOTS WITH REGRESSION LINES
# ============================================================================

def plot_key_relationships(df: pd.DataFrame) -> None:
    """
    2x2 scatter plots showing key significant relationships.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Churn vs Technical Debt
    ax = axes[0, 0]
    subset = df[df["sonar_sqale_index"].notna() & (df["churn_12m"] > 0)]
    if len(subset) > 10:
        ax.scatter(subset["log_churn"], subset["log_debt"], 
                   alpha=0.3, s=20, c=COLORS["primary"])
        # Add regression line
        z = np.polyfit(subset["log_churn"], subset["log_debt"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["log_churn"].min(), subset["log_churn"].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2, 
                label=f"β = 0.211***")
        ax.legend()
    ax.set_xlabel("Log(Churn + 1)")
    ax.set_ylabel("Log(Technical Debt + 1)")
    ax.set_title("A) Code Churn → Technical Debt", fontweight="bold")
    
    # Plot 2: Dominant Author Share vs Complexity
    ax = axes[0, 1]
    subset = df[df["dominant_author_share"].notna() & (df["dominant_author_share"] > 0)]
    if len(subset) > 10:
        ax.scatter(subset["dominant_author_share"], subset["log_complexity"],
                   alpha=0.3, s=20, c=COLORS["secondary"])
        z = np.polyfit(subset["dominant_author_share"], subset["log_complexity"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2,
                label=f"β = 0.391***")
        ax.legend()
    ax.set_xlabel("Dominant Author Share (0-1)")
    ax.set_ylabel("Log(Cyclomatic Complexity + 1)")
    ax.set_title("B) Knowledge Concentration → Complexity", fontweight="bold")
    
    # Plot 3: Unique Authors vs Cognitive Complexity
    ax = axes[1, 0]
    subset = df[df["unique_authors_12m"].notna() & df["sonar_cognitive_complexity"].notna()]
    if len(subset) > 10:
        # Jitter for discrete x values
        jitter = np.random.normal(0, 0.1, len(subset))
        ax.scatter(subset["unique_authors_12m"] + jitter, subset["log_cognitive"],
                   alpha=0.3, s=20, c=COLORS["tertiary"])
        z = np.polyfit(subset["unique_authors_12m"], subset["log_cognitive"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, subset["unique_authors_12m"].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2,
                label=f"β = 0.051***")
        ax.legend()
    ax.set_xlabel("Unique Authors (12 months)")
    ax.set_ylabel("Log(Cognitive Complexity + 1)")
    ax.set_title("C) Contributor Count → Cognitive Complexity", fontweight="bold")
    
    # Plot 4: PARTIAL RESIDUAL PLOT - Duplication vs Churn (both residualized for size)
    ax = axes[1, 1]
    subset = df[df["sonar_duplicated_lines_density"].notna() & 
                (df["churn_12m"] > 0) & 
                df["log_ncloc"].notna()].copy()
    if len(subset) > 10:
        # Residualize both variables against file size
        subset["churn_resid"] = residualize(subset["log_churn"], subset["log_ncloc"])
        subset["dup_resid"] = residualize(subset["sonar_duplicated_lines_density"], subset["log_ncloc"])
        
        valid = subset["churn_resid"].notna() & subset["dup_resid"].notna()
        ax.scatter(subset.loc[valid, "churn_resid"], subset.loc[valid, "dup_resid"],
                   alpha=0.4, s=20, c=COLORS["primary"])
        
        # Regression on residuals
        z = np.polyfit(subset.loc[valid, "churn_resid"], subset.loc[valid, "dup_resid"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["churn_resid"].min(), subset["churn_resid"].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2,
                label=f"β = 6.11** (size-adjusted)")
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Churn Residual (size-adjusted)")
    ax.set_ylabel("Duplication Residual (size-adjusted)")
    ax.set_title("D) Churn → Duplication (Partial Residuals)", fontweight="bold")
    
    plt.suptitle("Process Factors vs Sustainability Outcomes (Significant Relationships)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_scatter_relationships.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig2_scatter_relationships.png")


# ============================================================================
# FIGURE 3: CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap showing correlations between process and outcome variables.
    """
    # Select relevant columns
    process_cols = [
        "unique_authors_12m", "churn_12m", "dominant_author_share",
        "single_contributor_12m", "last_12m_observed_commits"
    ]
    outcome_cols = [
        "sonar_sqale_index", "sonar_code_smells", "sonar_complexity",
        "sonar_cognitive_complexity", "sonar_duplicated_lines_density"
    ]
    control_cols = ["sonar_ncloc"]
    
    all_cols = process_cols + outcome_cols + control_cols
    available_cols = [c for c in all_cols if c in df.columns]
    
    corr_df = df[available_cols].corr(method="spearman")
    
    # Create nicer labels
    label_map = {
        "unique_authors_12m": "Unique Authors",
        "churn_12m": "Code Churn",
        "dominant_author_share": "Dominant Author %",
        "single_contributor_12m": "Single Contributor",
        "last_12m_observed_commits": "Commit Count",
        "sonar_sqale_index": "Tech Debt (min)",
        "sonar_code_smells": "Code Smells",
        "sonar_complexity": "Cyclomatic Complexity",
        "sonar_cognitive_complexity": "Cognitive Complexity",
        "sonar_duplicated_lines_density": "Duplication %",
        "sonar_ncloc": "Lines of Code",
    }
    
    corr_df = corr_df.rename(index=label_map, columns=label_map)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman ρ"},
        ax=ax
    )
    
    ax.set_title("Bivariate Correlations: Process Factors & Sustainability Outcomes\n(Spearman's ρ — UNCONTROLLED, descriptive only)",
                 fontsize=13, fontweight="bold", pad=20)
    
    # Add clarifying note
    ax.text(0.5, -0.12, "Note: These are raw correlations. Regression models (Fig 1) control for file size and cluster by repository.",
            transform=ax.transAxes, fontsize=9, style='italic', color='gray', ha='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig3_correlation_heatmap.png")


# ============================================================================
# FIGURE 4: BOX PLOTS BY CONTRIBUTOR STATUS
# ============================================================================

def plot_single_vs_multi_contributor(df: pd.DataFrame) -> None:
    """
    Box plots comparing sustainability metrics for single vs multi-contributor files.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Prepare data - handle boolean values directly
    df = df.copy()
    df["contributor_type"] = df["single_contributor_12m"].apply(
        lambda x: "Single\nContributor" if x == True else ("Multiple\nContributors" if x == False else None)
    )
    df = df[df["contributor_type"].notna()]
    
    # Define order for consistent display
    order = ["Single\nContributor", "Multiple\nContributors"]
    
    # Plot 1: Technical Debt
    ax = axes[0]
    subset = df[df["sonar_sqale_index"].notna() & (df["sonar_sqale_index"] > 0)]
    if len(subset) > 10:
        sns.boxplot(
            data=subset,
            x="contributor_type",
            y="sonar_sqale_index",
            order=order,
            palette=[COLORS["primary"], COLORS["secondary"]],
            ax=ax
        )
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Technical Debt (minutes, log scale)")
    ax.set_title("A) Technical Debt", fontweight="bold")
    
    # Plot 2: Complexity
    ax = axes[1]
    subset = df[df["sonar_complexity"].notna() & (df["sonar_complexity"] > 0)]
    if len(subset) > 10:
        sns.boxplot(
            data=subset,
            x="contributor_type", 
            y="sonar_complexity",
            order=order,
            palette=[COLORS["primary"], COLORS["secondary"]],
            ax=ax
        )
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Cyclomatic Complexity (log scale)")
    ax.set_title("B) Cyclomatic Complexity", fontweight="bold")
    
    # Plot 3: Cognitive Complexity
    ax = axes[2]
    subset = df[df["sonar_cognitive_complexity"].notna() & (df["sonar_cognitive_complexity"] > 0)]
    if len(subset) > 10:
        sns.boxplot(
            data=subset,
            x="contributor_type",
            y="sonar_cognitive_complexity", 
            order=order,
            palette=[COLORS["primary"], COLORS["secondary"]],
            ax=ax
        )
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Cognitive Complexity (log scale)")
    ax.set_title("C) Cognitive Complexity", fontweight="bold")
    
    plt.suptitle("Sustainability Metrics by Contributor Pattern",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_contributor_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig4_contributor_comparison.png")


# ============================================================================
# FIGURE 5: FILE SIZE AS CONFOUNDER
# ============================================================================

def plot_size_confounder(df: pd.DataFrame) -> None:
    """
    Demonstrate that file size is the primary confounder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Size vs Tech Debt
    ax = axes[0]
    subset = df[df["sonar_sqale_index"].notna() & (df["sonar_ncloc"] > 0)]
    if len(subset) > 10:
        ax.scatter(subset["log_ncloc"], subset["log_debt"], 
                   alpha=0.2, s=10, c=COLORS["neutral"])
        z = np.polyfit(subset["log_ncloc"], subset["log_debt"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["log_ncloc"].min(), subset["log_ncloc"].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2.5)
        
        # Calculate R²
        r, _ = stats.pearsonr(subset["log_ncloc"], subset["log_debt"])
        ax.text(0.05, 0.95, f"R² = {r**2:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top", fontweight="bold")
    ax.set_xlabel("Log(Lines of Code)")
    ax.set_ylabel("Log(Technical Debt)")
    ax.set_title("A) File Size → Technical Debt", fontweight="bold")
    
    # Plot 2: Size vs Complexity
    ax = axes[1]
    subset = df[(df["sonar_ncloc"] > 0) & df["sonar_complexity"].notna()]
    if len(subset) > 10:
        ax.scatter(subset["log_ncloc"], subset["log_complexity"],
                   alpha=0.2, s=10, c=COLORS["neutral"])
        z = np.polyfit(subset["log_ncloc"], subset["log_complexity"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["log_ncloc"].min(), subset["log_ncloc"].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS["quaternary"], linewidth=2.5)
        
        r, _ = stats.pearsonr(subset["log_ncloc"], subset["log_complexity"])
        ax.text(0.05, 0.95, f"R² = {r**2:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top", fontweight="bold")
    ax.set_xlabel("Log(Lines of Code)")
    ax.set_ylabel("Log(Cyclomatic Complexity)")
    ax.set_title("B) File Size → Complexity", fontweight="bold")
    
    plt.suptitle("File Size as Primary Confounder (Controlled in All Models)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_size_confounder.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig5_size_confounder.png")


# ============================================================================
# FIGURE 6: REPOSITORY COMPARISON
# ============================================================================

def plot_repo_comparison(df: pd.DataFrame) -> None:
    """
    Compare sustainability metrics across repositories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aggregate by repo
    repo_stats = df.groupby("repo").agg({
        "sonar_sqale_index": "median",
        "sonar_complexity": "median",
        "churn_12m": "median",
        "unique_authors_12m": "median",
        "sonar_ncloc": "count"  # file count
    }).rename(columns={"sonar_ncloc": "file_count"})
    
    repo_stats = repo_stats.sort_values("sonar_sqale_index", ascending=False)
    
    # Plot 1: Median Technical Debt by Repo
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(repo_stats)))
    bars = ax.barh(repo_stats.index, repo_stats["sonar_sqale_index"], color=colors)
    ax.set_xlabel("Median Technical Debt (minutes)")
    ax.set_title("A) Technical Debt by Repository", fontweight="bold")
    ax.invert_yaxis()
    
    # Plot 2: Median Churn by Repo
    ax = axes[1]
    repo_stats_churn = repo_stats.sort_values("churn_12m", ascending=False)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(repo_stats_churn)))
    ax.barh(repo_stats_churn.index, repo_stats_churn["churn_12m"], color=colors)
    ax.set_xlabel("Median Code Churn (12 months)")
    ax.set_title("B) Code Churn by Repository", fontweight="bold")
    ax.invert_yaxis()
    
    plt.suptitle("Repository-Level Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_repo_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig6_repo_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    df = load_merged_data()
    print(f"Loaded {len(df)} records\n")
    
    # Generate all figures
    plot_coefficient_summary(df)
    plot_key_relationships(df)
    plot_correlation_heatmap(df)
    plot_single_vs_multi_contributor(df)
    plot_size_confounder(df)
    plot_repo_comparison(df)
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
