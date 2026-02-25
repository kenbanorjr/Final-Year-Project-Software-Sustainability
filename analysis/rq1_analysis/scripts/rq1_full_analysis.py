"""
RQ1 full analysis: alignment between SonarQube and LLM metrics.

Reads sonar_metrics.csv + llm_metrics_<model>_runNNN.csv (+ git_metrics.csv for language),
computes required tables/figures, and writes rq1_analysis outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.configs import config

SONAR_DEFAULT = config.sonar_metrics_path()
LLM_DEFAULT = config.llm_metrics_path(prefer_existing=True)
GIT_DEFAULT = config.git_metrics_path()

TABLES_DIR = config.BASE_DIR / "analysis" / "rq1_analysis" / "tables"
FIGURES_DIR = config.BASE_DIR / "analysis" / "rq1_analysis" / "figures"
SUMMARY_PATH = config.BASE_DIR / "analysis" / "rq1_analysis" / "RQ1_Summary.md"

LLM_COLOR = "#1f77b4"
SONAR_COLOR = "#ff7f0e"

METRIC_PAIRS = [
    ("sonar_complexity", "llm_cyclomatic_complexity", "Complexity"),
    ("sonar_cognitive_complexity", "llm_cognitive_complexity", "Cognitive Complexity"),
    ("sonar_ncloc", "llm_ncloc", "NCLOC"),
    ("sonar_comment_lines_density", "llm_comment_density", "Comment Density"),
    ("sonar_code_smells", "llm_code_smells", "Code Smells"),
    ("sonar_duplicated_lines_density", "llm_duplicated_lines_density", "Duplication"),
]

TECH_DEBT_PAIR = ("sonar_sqale_index", "llm_technical_debt_minutes", "Technical Debt")
RATING_PAIR = ("sonar_sqale_rating_num", "llm_maintainability_rating", "Maintainability Rating")

LANGUAGE_ORDER = ["python", "java", "javascript", "go", "typescript"]
LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "java": "java",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "go": "go",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ1 tables and figures.")
    parser.add_argument("--sonar", default=str(SONAR_DEFAULT), help="Path to sonar_metrics.csv")
    parser.add_argument("--llm", default=str(LLM_DEFAULT), help="Path to llm_metrics_<model>_runNNN.csv")
    parser.add_argument("--git", default=str(GIT_DEFAULT), help="Path to git_metrics.csv")
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_language(value: str) -> str:
    if not value:
        return "other"
    lowered = value.strip().lower()
    return LANGUAGE_ALIASES.get(lowered, lowered)


def _infer_language_from_path(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return LANGUAGE_ALIASES.get(ext, "other")


def _convert_sonar_rating(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        letter = value.strip().upper()
        if letter in {"A", "B", "C", "D", "E"}:
            return float({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}[letter])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metrics(sonar_path: Path, llm_path: Path, git_path: Path) -> pd.DataFrame:
    sonar_df = pd.read_csv(sonar_path)
    llm_df = pd.read_csv(llm_path)
    git_df = pd.read_csv(git_path) if git_path.exists() else pd.DataFrame()

    for df in (sonar_df, llm_df, git_df):
        if not df.empty:
            df["file_path"] = df["file_path"].astype(str).str.replace("\\", "/", regex=False)

    if "llm_success" in llm_df.columns:
        llm_df = llm_df[llm_df["llm_success"].astype(str).str.lower().isin({"true", "1", "yes"})]

    sonar_df = sonar_df.drop_duplicates(subset=["repo", "file_path"])
    llm_df = llm_df.drop_duplicates(subset=["repo", "file_path"], keep="last")

    merged = sonar_df.merge(llm_df, on=["repo", "file_path"], how="inner", suffixes=("", "_llm"))

    if not git_df.empty and {"repo", "file_path", "file_language"}.issubset(git_df.columns):
        git_df = git_df[["repo", "file_path", "file_language"]].drop_duplicates(
            subset=["repo", "file_path"]
        )
        merged = merged.merge(git_df, on=["repo", "file_path"], how="left")
        merged["file_language"] = merged["file_language"].fillna("")
    else:
        merged["file_language"] = ""

    merged["file_language"] = merged["file_language"].apply(_normalize_language)
    missing_lang = merged["file_language"].eq("") | merged["file_language"].eq("other")
    merged.loc[missing_lang, "file_language"] = merged.loc[missing_lang, "file_path"].map(
        _infer_language_from_path
    )

    merged["sonar_sqale_rating_num"] = merged["sonar_sqale_rating"].apply(_convert_sonar_rating)
    merged["llm_maintainability_rating"] = pd.to_numeric(
        merged.get("llm_maintainability_rating"), errors="coerce"
    )

    numeric_cols = [
        "sonar_complexity",
        "sonar_cognitive_complexity",
        "sonar_ncloc",
        "sonar_comment_lines_density",
        "sonar_code_smells",
        "sonar_duplicated_lines_density",
        "sonar_sqale_index",
        "sonar_vulnerabilities",
        "llm_cyclomatic_complexity",
        "llm_cognitive_complexity",
        "llm_ncloc",
        "llm_comment_density",
        "llm_code_smells",
        "llm_duplicated_lines_density",
        "llm_technical_debt_minutes",
        "llm_security_issues",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


def _interpret_correlation(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 0.7:
        strength = "strong"
    elif magnitude >= 0.4:
        strength = "moderate"
    elif magnitude >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    direction = "positive" if value >= 0 else "negative"
    return f"{strength} {direction}"


def _has_variance(series: pd.Series) -> bool:
    return series.dropna().nunique() > 1


def _safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    if not _has_variance(x) or not _has_variance(y):
        return None, None
    r, p = stats.pearsonr(x, y)
    if math.isnan(r):
        return None, None
    return float(r), float(p)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    if not _has_variance(x) or not _has_variance(y):
        return None, None
    rho, p = stats.spearmanr(x, y)
    if math.isnan(rho):
        return None, None
    return float(rho), float(p)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sonar_col, llm_col, label in METRIC_PAIRS + [TECH_DEBT_PAIR]:
        subset = df[[sonar_col, llm_col]].dropna()
        n_samples = len(subset)
        if n_samples < 2:
            rows.append(
                {
                    "Metric_Pair": label,
                    "Pearson_r": None,
                    "Pearson_p_value": None,
                    "Spearman_rho": None,
                    "Spearman_p_value": None,
                    "N_samples": n_samples,
                    "Interpretation": "n/a",
                }
            )
            continue
        pearson_r, pearson_p = _safe_pearson(subset[sonar_col], subset[llm_col])
        spearman_rho, spearman_p = _safe_spearman(subset[sonar_col], subset[llm_col])
        rows.append(
            {
                "Metric_Pair": label,
                "Pearson_r": pearson_r,
                "Pearson_p_value": pearson_p,
                "Spearman_rho": spearman_rho,
                "Spearman_p_value": spearman_p,
                "N_samples": n_samples,
                "Interpretation": _interpret_correlation(spearman_rho),
            }
        )
    return pd.DataFrame(rows)


def rating_agreement(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df[[RATING_PAIR[0], RATING_PAIR[1]]].dropna()
    if ratings.empty:
        return pd.DataFrame(
            [
                {
                    "Rating_Type": RATING_PAIR[2],
                    "Cohens_Kappa": None,
                    "Percent_Exact_Agreement": None,
                    "Percent_Within_1": None,
                    "Percent_Within_2": None,
                    "Weighted_Kappa": None,
                }
            ]
        )
    sonar = ratings[RATING_PAIR[0]].round().astype(int)
    llm = ratings[RATING_PAIR[1]].round().astype(int)
    mask = sonar.between(1, 5) & llm.between(1, 5)
    sonar = sonar[mask]
    llm = llm[mask]
    diff = (sonar - llm).abs()
    kappa = cohen_kappa_score(sonar, llm)
    weighted = cohen_kappa_score(sonar, llm, weights="quadratic")
    return pd.DataFrame(
        [
            {
                "Rating_Type": RATING_PAIR[2],
                "Cohens_Kappa": kappa,
                "Percent_Exact_Agreement": (diff == 0).mean() * 100.0,
                "Percent_Within_1": (diff <= 1).mean() * 100.0,
                "Percent_Within_2": (diff <= 2).mean() * 100.0,
                "Weighted_Kappa": weighted,
            }
        ]
    )


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sonar_col, llm_col, label in METRIC_PAIRS + [TECH_DEBT_PAIR]:
        rows.append(_describe_series(df[sonar_col], label, "Sonar"))
        rows.append(_describe_series(df[llm_col], label, "LLM"))
    return pd.DataFrame(rows)


def _describe_series(series: pd.Series, label: str, source: str) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return {
            "Metric": label,
            "Source": source,
            "Mean": None,
            "Median": None,
            "Std_Dev": None,
            "Min": None,
            "Max": None,
            "Q1": None,
            "Q3": None,
            "N": 0,
        }
    return {
        "Metric": label,
        "Source": source,
        "Mean": series.mean(),
        "Median": series.median(),
        "Std_Dev": series.std(ddof=1),
        "Min": series.min(),
        "Max": series.max(),
        "Q1": series.quantile(0.25),
        "Q3": series.quantile(0.75),
        "N": int(series.count()),
    }


def disagreement_analysis(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []

    complexity_q25 = df["sonar_complexity"].quantile(0.25)
    complexity_q75 = df["sonar_complexity"].quantile(0.75)
    llm_complexity_q25 = df["llm_cyclomatic_complexity"].quantile(0.25)
    llm_complexity_q75 = df["llm_cyclomatic_complexity"].quantile(0.75)

    high_llm_low_sonar = df[
        (df["llm_cyclomatic_complexity"] >= llm_complexity_q75)
        & (df["sonar_complexity"] <= complexity_q25)
    ]
    low_llm_high_sonar = df[
        (df["llm_cyclomatic_complexity"] <= llm_complexity_q25)
        & (df["sonar_complexity"] >= complexity_q75)
    ]

    rating_diff = (df[RATING_PAIR[0]] - df[RATING_PAIR[1]]).abs()
    rating_diff_2plus = df[rating_diff >= 2]

    smell_diff = (df["sonar_code_smells"] - df["llm_code_smells"]).abs()
    smell_threshold = max(5.0, smell_diff.quantile(0.75))
    smell_major = df[smell_diff >= smell_threshold]

    rows.append(
        _disagreement_row(
            "High_LLM_Low_Sonar_Complexity", high_llm_low_sonar, total, "llm_cyclomatic_complexity"
        )
    )
    rows.append(
        _disagreement_row(
            "Low_LLM_High_Sonar_Complexity", low_llm_high_sonar, total, "sonar_complexity"
        )
    )
    rows.append(
        _disagreement_row(
            "Rating_Difference_2plus", rating_diff_2plus, total, "sonar_sqale_rating_num"
        )
    )
    rows.append(
        _disagreement_row(
            "Major_Code_Smell_Disagreement", smell_major, total, "sonar_code_smells"
        )
    )
    return pd.DataFrame(rows)


def _disagreement_row(name: str, subset: pd.DataFrame, total: int, sort_col: str) -> dict:
    subset = subset.copy()
    examples = []
    if not subset.empty:
        subset = subset.sort_values(sort_col, ascending=False)
        for row in subset.itertuples():
            examples.append(f"{row.repo}/{row.file_path}")
            if len(examples) >= 5:
                break
    return {
        "Disagreement_Type": name,
        "Count": len(subset),
        "Percent_of_Total": (len(subset) / total * 100.0) if total else 0.0,
        "Top_5_Examples": "; ".join(examples),
    }


def per_language_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for language in LANGUAGE_ORDER:
        subset = df[df["file_language"] == language]
        rows.append(_alignment_row(subset, language))
    return pd.DataFrame(rows)


def per_repository_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repo, subset in df.groupby("repo"):
        rows.append(_repo_alignment_row(subset, repo))
    return pd.DataFrame(rows)


def _alignment_row(subset: pd.DataFrame, label: str) -> dict:
    n_files = len(subset)
    complexity_corr = _spearman(subset, "sonar_complexity", "llm_cyclomatic_complexity")
    cognitive_corr = _spearman(subset, "sonar_cognitive_complexity", "llm_cognitive_complexity")
    ncloc_corr = _spearman(subset, "sonar_ncloc", "llm_ncloc")
    kappa = _kappa(subset)
    overall = _overall_alignment([complexity_corr, cognitive_corr, ncloc_corr], kappa)
    return {
        "Language": label,
        "N_Files": n_files,
        "Complexity_Correlation": complexity_corr,
        "Cognitive_Correlation": cognitive_corr,
        "NCLOC_Correlation": ncloc_corr,
        "Maintainability_Kappa": kappa,
        "Overall_Alignment_Score": overall,
    }


def _repo_alignment_row(subset: pd.DataFrame, repo: str) -> dict:
    n_files = len(subset)
    complexity_corr = _spearman(subset, "sonar_complexity", "llm_cyclomatic_complexity")
    code_smells_corr = _spearman(subset, "sonar_code_smells", "llm_code_smells")
    kappa = _kappa(subset)
    overall = _overall_alignment([complexity_corr, code_smells_corr], kappa)
    return {
        "Repository": repo,
        "N_Files": n_files,
        "Mean_Complexity_Correlation": complexity_corr,
        "Maintainability_Kappa": kappa,
        "Code_Smells_Correlation": code_smells_corr,
        "Overall_Alignment_Score": overall,
    }


def _overall_alignment(corrs: Iterable[float | None], kappa: float | None) -> float | None:
    values = []
    for value in corrs:
        if value is not None and not math.isnan(value):
            values.append(abs(value))
    if kappa is not None and not math.isnan(kappa):
        values.append((kappa + 1.0) / 2.0)
    if not values:
        return None
    return float(np.mean(values))


def _spearman(df: pd.DataFrame, sonar_col: str, llm_col: str) -> float | None:
    subset = df[[sonar_col, llm_col]].dropna()
    if len(subset) < 2:
        return None
    rho, _ = _safe_spearman(subset[sonar_col], subset[llm_col])
    return rho


def _kappa(df: pd.DataFrame) -> float | None:
    ratings = df[[RATING_PAIR[0], RATING_PAIR[1]]].dropna()
    if len(ratings) < 2:
        return None
    sonar = ratings[RATING_PAIR[0]].round().astype(int)
    llm = ratings[RATING_PAIR[1]].round().astype(int)
    mask = sonar.between(1, 5) & llm.between(1, 5)
    if mask.sum() < 2:
        return None
    return float(cohen_kappa_score(sonar[mask], llm[mask]))


def confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    ratings = df[[RATING_PAIR[0], RATING_PAIR[1]]].dropna()
    sonar = ratings[RATING_PAIR[0]].round().astype(int)
    llm = ratings[RATING_PAIR[1]].round().astype(int)
    mask = sonar.between(1, 5) & llm.between(1, 5)
    matrix = pd.crosstab(llm[mask], sonar[mask])
    matrix = matrix.reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5], fill_value=0)
    return matrix


def scatter_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300, constrained_layout=True)
    palette = sns.color_palette("tab10", len(LANGUAGE_ORDER) + 1)
    lang_palette = {lang: palette[i] for i, lang in enumerate(LANGUAGE_ORDER)}
    lang_palette["other"] = palette[-1]

    for ax, (sonar_col, llm_col, label) in zip(axes.flat, METRIC_PAIRS):
        subset = df[[sonar_col, llm_col, "file_language"]].dropna()
        if subset.empty:
            ax.set_title(label)
            ax.axis("off")
            continue
        sns.scatterplot(
            data=subset,
            x=sonar_col,
            y=llm_col,
            hue="file_language",
            palette=lang_palette,
            ax=ax,
            alpha=0.6,
            s=18,
            linewidth=0,
        )
        sns.regplot(
            data=subset,
            x=sonar_col,
            y=llm_col,
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linewidth": 1},
        )
        min_val = min(subset[sonar_col].min(), subset[llm_col].min())
        max_val = max(subset[sonar_col].max(), subset[llm_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
        r, _ = _safe_pearson(subset[sonar_col], subset[llm_col])
        r2_label = "n/a" if r is None else f"{r**2:.2f}"
        ax.text(
            0.02,
            0.95,
            f"$R^2$={r2_label}\\nN={len(subset)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set_title(label)
        ax.set_xlabel("Sonar")
        ax.set_ylabel("LLM")
        ax.legend_.remove()

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(LANGUAGE_ORDER))
    fig.suptitle("LLM vs Sonar Metrics (Scatter + Regression)", fontsize=12, y=1.05)
    fig.savefig(FIGURES_DIR / "scatter_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame) -> None:
    sonar_metrics = [
        "sonar_complexity",
        "sonar_cognitive_complexity",
        "sonar_ncloc",
        "sonar_comment_lines_density",
        "sonar_code_smells",
        "sonar_duplicated_lines_density",
        "sonar_sqale_index",
    ]
    llm_metrics = [
        "llm_cyclomatic_complexity",
        "llm_cognitive_complexity",
        "llm_ncloc",
        "llm_comment_density",
        "llm_code_smells",
        "llm_duplicated_lines_density",
        "llm_technical_debt_minutes",
    ]
    matrix = pd.DataFrame(index=llm_metrics, columns=sonar_metrics, dtype=float)
    for llm_col in llm_metrics:
        for sonar_col in sonar_metrics:
            subset = df[[sonar_col, llm_col]].dropna()
            if len(subset) < 2 or not _has_variance(subset[sonar_col]) or not _has_variance(subset[llm_col]):
                matrix.loc[llm_col, sonar_col] = np.nan
            else:
                rho, _ = _safe_spearman(subset[sonar_col], subset[llm_col])
                matrix.loc[llm_col, sonar_col] = rho
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
    )
    plt.title("Correlation Heatmap (LLM vs Sonar)")
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def rating_confusion_matrix(df: pd.DataFrame) -> None:
    matrix = confusion_matrix(df)
    total = matrix.values.sum()
    percent = matrix / total * 100 if total else matrix
    labels = matrix.astype(int).astype(str) + "\n" + percent.round(1).astype(str) + "%"
    plt.figure(figsize=(8, 8), dpi=300)
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
    )
    plt.xlabel("Sonar Maintainability Rating")
    plt.ylabel("LLM Maintainability Rating")
    plt.title("Maintainability Rating Confusion Matrix")
    plt.savefig(FIGURES_DIR / "rating_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def distribution_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300, constrained_layout=True)
    for ax, (sonar_col, llm_col, label) in zip(axes.flat, METRIC_PAIRS):
        subset = df[[sonar_col, llm_col]].dropna()
        data = pd.DataFrame(
            {
                "value": pd.concat([subset[sonar_col], subset[llm_col]], ignore_index=True),
                "source": ["Sonar"] * len(subset) + ["LLM"] * len(subset),
            }
        )
        sns.boxplot(
            data=data,
            x="source",
            y="value",
            hue="source",
            ax=ax,
            palette={"Sonar": SONAR_COLOR, "LLM": LLM_COLOR},
            legend=False,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Distribution Comparison (LLM vs Sonar)", fontsize=12)
    fig.savefig(FIGURES_DIR / "distribution_comparison_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bland_altman_plots(df: pd.DataFrame) -> None:
    pairs = [
        ("sonar_complexity", "llm_cyclomatic_complexity", "Complexity"),
        ("sonar_cognitive_complexity", "llm_cognitive_complexity", "Cognitive Complexity"),
        ("sonar_ncloc", "llm_ncloc", "NCLOC"),
        ("sonar_code_smells", "llm_code_smells", "Code Smells"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300, constrained_layout=True)
    for ax, (sonar_col, llm_col, label) in zip(axes.flat, pairs):
        subset = df[[sonar_col, llm_col]].dropna()
        if subset.empty:
            ax.axis("off")
            continue
        mean = (subset[sonar_col] + subset[llm_col]) / 2.0
        diff = subset[llm_col] - subset[sonar_col]
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)
        ax.scatter(mean, diff, alpha=0.5, s=16, color=LLM_COLOR)
        ax.axhline(mean_diff, color="black", linestyle="--", linewidth=1)
        ax.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle="--", linewidth=1)
        ax.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Mean of LLM and Sonar")
        ax.set_ylabel("LLM - Sonar")
    fig.suptitle("Bland-Altman Plots")
    fig.savefig(FIGURES_DIR / "bland_altman_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def agreement_rates_bar(df: pd.DataFrame) -> None:
    rows = []
    for label, sonar_col, llm_col in [
        ("complexity", "sonar_complexity", "llm_cyclomatic_complexity"),
        ("code_smells", "sonar_code_smells", "llm_code_smells"),
    ]:
        subset = df[[sonar_col, llm_col]].dropna()
        if subset.empty:
            continue
        z_sonar = (subset[sonar_col] - subset[sonar_col].mean()) / subset[sonar_col].std(ddof=1)
        z_llm = (subset[llm_col] - subset[llm_col].mean()) / subset[llm_col].std(ddof=1)
        diff = (z_llm - z_sonar).abs()
        rows.extend(_agreement_buckets(label, diff))

    rating_diff = (df[RATING_PAIR[0]] - df[RATING_PAIR[1]]).abs().dropna()
    if not rating_diff.empty:
        rows.extend(_agreement_buckets("ratings", rating_diff))

    data = pd.DataFrame(rows)
    # Mutually exclusive bins - each sums to 100%
    order = ["Exact Match", "Off by 1", "Off by 2", "Off by >2"]
    plt.figure(figsize=(10, 6), dpi=300)
    pivot = data.pivot_table(
        index="Metric", columns="Agreement_Level", values="Percent", aggfunc="sum"
    ).reindex(columns=order)
    pivot.plot(
        kind="bar",
        stacked=True,
        color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"],
        figsize=(10, 6),
    )
    plt.ylabel("Percent of Files")
    plt.title("Agreement Rates by Metric Category (Mutually Exclusive Bins)")
    plt.legend(title="Agreement Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "agreement_rates_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def _agreement_buckets(metric_label: str, diff: pd.Series) -> list[dict]:
    """
    Create MUTUALLY EXCLUSIVE agreement buckets.
    Each file falls into exactly one bucket, so totals sum to 100%.
    """
    buckets = {
        "Exact Match": (diff == 0).mean() * 100.0,
        "Off by 1": ((diff > 0) & (diff <= 1)).mean() * 100.0,
        "Off by 2": ((diff > 1) & (diff <= 2)).mean() * 100.0,
        "Off by >2": (diff > 2).mean() * 100.0,
    }
    return [
        {"Metric": metric_label, "Agreement_Level": level, "Percent": pct}
        for level, pct in buckets.items()
    ]


def per_language_comparison(language_table: pd.DataFrame) -> None:
    plot_df = language_table.melt(
        id_vars=["Language"],
        value_vars=["Complexity_Correlation", "Cognitive_Correlation", "NCLOC_Correlation"],
        var_name="Metric",
        value_name="Correlation",
    )
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=plot_df,
        x="Language",
        y="Correlation",
        hue="Metric",
        order=LANGUAGE_ORDER,
    )
    plt.title("Per-Language Correlations")
    plt.ylabel("Spearman Correlation")
    plt.xlabel("Language")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_language_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def density_overlaps(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300, constrained_layout=True)
    for ax, (sonar_col, llm_col, label) in zip(axes.flat, METRIC_PAIRS):
        subset = df[[sonar_col, llm_col]].dropna()
        if subset.empty:
            ax.axis("off")
            continue
        sns.kdeplot(subset[sonar_col], ax=ax, color=SONAR_COLOR, label="Sonar", fill=True, alpha=0.3)
        sns.kdeplot(subset[llm_col], ax=ax, color=LLM_COLOR, label="LLM", fill=True, alpha=0.3)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.legend()
    fig.suptitle("Density Overlaps (LLM vs Sonar)")
    fig.savefig(FIGURES_DIR / "density_overlaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def per_repository_heatmap(repo_table: pd.DataFrame) -> None:
    metrics = ["Mean_Complexity_Correlation", "Code_Smells_Correlation", "Maintainability_Kappa"]
    heatmap_df = repo_table.set_index("Repository")[metrics]
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(heatmap_df, cmap="RdBu_r", vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title("Per-Repository Alignment Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_repository_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def residual_plots(df: pd.DataFrame) -> None:
    pairs = [
        ("sonar_complexity", "llm_cyclomatic_complexity", "Complexity"),
        ("sonar_cognitive_complexity", "llm_cognitive_complexity", "Cognitive Complexity"),
        ("sonar_ncloc", "llm_ncloc", "NCLOC"),
        ("sonar_code_smells", "llm_code_smells", "Code Smells"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300, constrained_layout=True)
    for ax, (sonar_col, llm_col, label) in zip(axes.flat, pairs):
        subset = df[[sonar_col, llm_col]].dropna()
        if len(subset) < 3:
            ax.axis("off")
            continue
        slope, intercept, _, _, _ = stats.linregress(subset[sonar_col], subset[llm_col])
        fitted = slope * subset[sonar_col] + intercept
        residuals = subset[llm_col] - fitted
        ax.scatter(fitted, residuals, s=16, alpha=0.6, color=LLM_COLOR)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        inset = ax.inset_axes([0.62, 0.55, 0.35, 0.35])
        stats.probplot(residuals, dist="norm", plot=inset)
        inset.set_title("Q-Q", fontsize=7)
        inset.tick_params(axis="both", labelsize=6)
    fig.suptitle("Residual Plots with Q-Q Insets")
    fig.savefig(FIGURES_DIR / "residual_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_summary(df: pd.DataFrame) -> dict:
    summary = {}
    summary["n_files"] = len(df)
    correlation_df = correlation_analysis(df)
    summary["median_spearman"] = correlation_df["Spearman_rho"].median()
    rating_df = rating_agreement(df)
    summary["maintainability_kappa"] = rating_df["Cohens_Kappa"].iloc[0]
    disagreement_df = disagreement_analysis(df)
    top_disagreement = disagreement_df.sort_values("Percent_of_Total", ascending=False).head(1)
    if not top_disagreement.empty:
        summary["top_disagreement"] = top_disagreement["Disagreement_Type"].iloc[0]
        summary["top_disagreement_pct"] = top_disagreement["Percent_of_Total"].iloc[0]
    else:
        summary["top_disagreement"] = "n/a"
        summary["top_disagreement_pct"] = None
    return summary


def write_summary(summary: dict) -> None:
    lines = [
        "# RQ1 Summary",
        "",
        f"- Files analyzed (LLM + Sonar overlap): {summary.get('n_files', 0)}",
        f"- Median Spearman correlation (paired metrics): {summary.get('median_spearman'):.3f}",
        f"- Maintainability Cohen's kappa: {summary.get('maintainability_kappa'):.3f}",
        f"- Top disagreement type: {summary.get('top_disagreement')} "
        f"({summary.get('top_disagreement_pct', 0):.2f}%)",
        "",
        "Notes:",
        "- Correlations use only rows with both Sonar and LLM metrics present.",
        "- Maintainability uses Sonar SQALE rating mapped to 1–5.",
        "- Agreement rate bars use standardized differences for continuous metrics.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    df = load_metrics(Path(args.sonar), Path(args.llm), Path(args.git))

    corr_table = correlation_analysis(df)
    corr_table.to_csv(TABLES_DIR / "correlation_analysis.csv", index=False)

    rating_table = rating_agreement(df)
    rating_table.to_csv(TABLES_DIR / "rating_agreement.csv", index=False)

    desc_table = descriptive_statistics(df)
    desc_table.to_csv(TABLES_DIR / "descriptive_statistics.csv", index=False)

    disagree_table = disagreement_analysis(df)
    disagree_table.to_csv(TABLES_DIR / "disagreement_analysis.csv", index=False)

    lang_table = per_language_analysis(df)
    lang_table.to_csv(TABLES_DIR / "per_language_analysis.csv", index=False)

    repo_table = per_repository_analysis(df)
    repo_table.to_csv(TABLES_DIR / "per_repository_analysis.csv", index=False)

    conf_matrix = confusion_matrix(df)
    conf_matrix.to_csv(TABLES_DIR / "confusion_matrix_maintainability.csv")

    scatter_plots(df)
    correlation_heatmap(df)
    rating_confusion_matrix(df)
    distribution_boxplots(df)
    bland_altman_plots(df)
    agreement_rates_bar(df)
    per_language_comparison(lang_table)
    density_overlaps(df)
    per_repository_heatmap(repo_table)
    residual_plots(df)

    summary = scatter_summary(df)
    write_summary(summary)

    print(f"[rq1] Wrote tables to {TABLES_DIR}")
    print(f"[rq1] Wrote figures to {FIGURES_DIR}")
    print(f"[rq1] Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
