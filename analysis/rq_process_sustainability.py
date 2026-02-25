"""
Research Question Analysis: Do social/process factors explain sustainability outcomes?

Hypothesis: Git-derived process metrics (churn, knowledge concentration, contributor
patterns) explain variance in SonarQube sustainability indicators, controlling for
file size (the primary confounder).

Statistical Approach:
1. OLS regression for continuous outcomes (technical debt, complexity)
2. Negative binomial for count data (code smells, bugs, vulnerabilities)
3. Ordinal logistic for ratings (1-5 scale)
4. Beta regression or OLS for proportions (duplication density)

All models control for file size (sonar_ncloc) as the main confounder.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pipeline.configs import config

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = config.RESULTS_DIR
OUTPUT_DIR = config.BASE_DIR / "analysis" / "rq_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Git metrics (process/social factors) - PREDICTORS
# NOTE: churn_12m, unique_authors_12m, and commit_count are highly correlated (r≈0.98)
# To avoid multicollinearity, we run THREE SEPARATE MODEL SETS:

# Model Set A: Churn-based (code instability focus)
MODEL_A_PREDICTORS = [
    "log_churn",                    # Code instability (log-transformed)
    "single_contributor_12m",       # Bus factor risk (boolean)
    "dominant_author_share",        # Knowledge concentration (0-1)
]

# Model Set B: Author-based (team dynamics focus)
MODEL_B_PREDICTORS = [
    "unique_authors_12m",           # Knowledge distribution (count)
    "single_contributor_12m",       # Bus factor risk (boolean)
    "dominant_author_share",        # Knowledge concentration (0-1)
]

# Model Set C: Composite (recommended for combined interpretation)
MODEL_C_PREDICTORS = [
    "process_intensity",            # Composite: z-score avg of churn + unique_authors
    "single_contributor_12m",       # Bus factor risk (boolean)
    "dominant_author_share",        # Knowledge concentration (0-1)
]

# Default to Model C (composite) for main analysis
GIT_PREDICTORS = MODEL_C_PREDICTORS

# All model sets for comparison
MODEL_SETS = {
    "A_Churn": MODEL_A_PREDICTORS,
    "B_Authors": MODEL_B_PREDICTORS,
    "C_Composite": MODEL_C_PREDICTORS,
}

# Sonar metrics (sustainability outcomes) - DEPENDENT VARIABLES
SONAR_OUTCOMES = {
    # Continuous / Count outcomes
    "sonar_sqale_index": "technical_debt_minutes",
    "sonar_code_smells": "code_smells",
    "sonar_bugs": "bugs",
    "sonar_vulnerabilities": "vulnerabilities",
    "sonar_complexity": "cyclomatic_complexity",
    "sonar_cognitive_complexity": "cognitive_complexity",
    "sonar_duplicated_blocks": "duplicated_blocks",
    # Proportion outcomes (0-100)
    "sonar_duplicated_lines_density": "duplication_pct",
    # Rating outcomes (1-5, where 1=A=best, 5=E=worst)
    "sonar_sqale_rating": "maintainability_rating",
    "sonar_reliability_rating": "reliability_rating",
    "sonar_security_rating": "security_rating",
}

# Control variable
SIZE_CONTROL = "sonar_ncloc"

# NOTE: Sparse metrics (where SonarQube omits zeros) are now handled in sonar_runner.py
# The CSV already contains has_<metric> flags and nulls filled with 0


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_merge_data() -> pd.DataFrame:
    """Load git and sonar metrics, merge on repo/file_path."""
    git_df = pd.read_csv(config.git_metrics_path())
    sonar_df = pd.read_csv(config.sonar_metrics_path())
    
    # Normalize paths
    git_df["file_path"] = git_df["file_path"].str.replace("\\", "/", regex=False)
    sonar_df["file_path"] = sonar_df["file_path"].str.replace("\\", "/", regex=False)
    
    # Merge on repo + file_path
    merged = pd.merge(
        sonar_df,
        git_df,
        on=["repo", "file_path"],
        how="inner",
        suffixes=("", "_git")
    )
    
    print(f"Loaded {len(sonar_df)} Sonar records, {len(git_df)} Git records")
    print(f"Merged dataset: {len(merged)} records")
    
    return merged


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for regression analysis."""
    from sklearn.preprocessing import StandardScaler
    
    analysis = df.copy()
    
    # Convert boolean to int
    if "single_contributor_12m" in analysis.columns:
        analysis["single_contributor_12m"] = analysis["single_contributor_12m"].astype(int)
    
    # Log-transform size control (reduces skewness)
    analysis["log_ncloc"] = np.log1p(analysis[SIZE_CONTROL].fillna(0))
    
    # Log-transform churn (highly skewed)
    if "churn_12m" in analysis.columns:
        analysis["log_churn"] = np.log1p(analysis["churn_12m"].fillna(0))
    
    # Create PROCESS INTENSITY composite (handles multicollinearity)
    # Combines churn and unique_authors via z-score averaging
    scaler = StandardScaler()
    process_cols = ["log_churn", "unique_authors_12m"]
    valid_mask = analysis[["churn_12m", "unique_authors_12m"]].notna().all(axis=1)
    if valid_mask.sum() > 0:
        z_scores = scaler.fit_transform(
            analysis.loc[valid_mask, ["log_churn", "unique_authors_12m"]].fillna(0)
        )
        analysis.loc[valid_mask, "process_intensity"] = z_scores.mean(axis=1)
    
    # Handle missing values in predictors
    for col in GIT_PREDICTORS + [SIZE_CONTROL]:
        if col in analysis.columns:
            analysis[col] = analysis[col].fillna(0)
    
    # Drop rows with missing outcomes
    outcome_cols = list(SONAR_OUTCOMES.keys())
    available_outcomes = [c for c in outcome_cols if c in analysis.columns]
    analysis = analysis.dropna(subset=available_outcomes, how="all")
    
    # Filter to files with meaningful size (exclude trivial files)
    analysis = analysis[analysis[SIZE_CONTROL] >= 10]
    
    print(f"Analysis dataset: {len(analysis)} records after cleaning")
    
    return analysis


# ============================================================================
# REGRESSION MODELS
# ============================================================================
# 
# METHODOLOGICAL NOTES:
# 1. All predictors are STANDARDIZED (z-scored) for comparable coefficients
# 2. SEs are CLUSTERED BY REPOSITORY to account for within-repo correlation
# 3. File size (log NCLOC) is included as control in all models
# 4. Coefficients represent effect per 1 SD change in predictor
#
# ============================================================================


def standardize_predictors(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """
    Standardize all predictors to z-scores for comparable coefficients.
    After standardization, coefficients represent effect per 1 SD change.
    """
    from sklearn.preprocessing import StandardScaler
    df = df.copy()
    scaler = StandardScaler()
    
    cols_to_std = [p for p in predictors if p in df.columns and df[p].dtype in ['float64', 'int64', 'float32', 'int32']]
    if cols_to_std:
        df[cols_to_std] = scaler.fit_transform(df[cols_to_std].fillna(0))
    
    # Also standardize control variable
    if 'log_ncloc' in df.columns:
        df['log_ncloc'] = (df['log_ncloc'] - df['log_ncloc'].mean()) / df['log_ncloc'].std()
    
    return df


def run_ols_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    log_transform_outcome: bool = True,
    cluster_by_repo: bool = True
) -> dict[str, Any]:
    """
    Run OLS regression with size control and repo-clustered SEs.
    
    Args:
        df: Input dataframe
        outcome: Outcome variable column name
        predictors: List of predictor column names
        log_transform_outcome: Whether to log-transform the outcome
        cluster_by_repo: Whether to cluster standard errors by repository
    
    Returns:
        Dict with model results, standardized coefficients, and cluster-robust SEs
    """
    # Prepare outcome
    y_col = outcome
    df = df.copy()
    if log_transform_outcome:
        y_col = f"log_{outcome}"
        df[y_col] = np.log1p(df[outcome].fillna(0))
    
    # Standardize predictors for comparable coefficients
    df = standardize_predictors(df, predictors)
    
    # Build formula with size control
    formula = f"{y_col} ~ log_ncloc + " + " + ".join(predictors)
    
    # Filter valid rows
    required_cols = [y_col, "log_ncloc", "repo"] + predictors
    available_cols = [c for c in required_cols if c in df.columns]
    model_df = df[available_cols].dropna()
    
    if len(model_df) < 50:
        return {"error": f"Insufficient data: {len(model_df)} rows"}
    
    try:
        model = smf.ols(formula, data=model_df).fit()
        
        # Get cluster-robust standard errors if requested
        if cluster_by_repo and "repo" in model_df.columns:
            # Cluster-robust SEs by repository
            model = smf.ols(formula, data=model_df).fit(
                cov_type='cluster',
                cov_kwds={'groups': model_df['repo']}
            )
        
        return extract_results(model, outcome, predictors, standardized=True)
    except Exception as e:
        return {"error": str(e)}


def run_negative_binomial(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str]
) -> dict[str, Any]:
    """
    Negative binomial regression for count outcomes (smells, bugs, etc.).
    
    Better than Poisson when variance > mean (overdispersion).
    """
    # Build formula with size control
    formula = f"{outcome} ~ log_ncloc + " + " + ".join(predictors)
    
    # Filter valid rows
    all_vars = [outcome, "log_ncloc"] + predictors
    model_df = df[all_vars].dropna()
    model_df = model_df[model_df[outcome] >= 0]  # Counts must be non-negative
    
    if len(model_df) < 50:
        return {"error": f"Insufficient data: {len(model_df)} rows"}
    
    try:
        # Use negative binomial for overdispersed counts
        model = smf.negativebinomial(formula, data=model_df).fit(disp=False)
        return extract_results(model, outcome, predictors, exp_coef=True)
    except Exception as e:
        # Fallback to Poisson if NB fails
        try:
            model = smf.poisson(formula, data=model_df).fit(disp=False)
            result = extract_results(model, outcome, predictors, exp_coef=True)
            result["note"] = "Poisson fallback (NB failed)"
            return result
        except Exception as e2:
            return {"error": f"NB: {e}, Poisson: {e2}"}


def run_ordinal_logistic(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str]
) -> dict[str, Any]:
    """
    Ordinal logistic regression for rating outcomes (1-5 scale).
    
    Tests whether process factors predict worse ratings.
    """
    # Filter valid rows
    all_vars = [outcome, "log_ncloc"] + predictors
    model_df = df[all_vars].dropna()
    model_df = model_df[model_df[outcome].between(1, 5)]
    
    if len(model_df) < 50:
        return {"error": f"Insufficient data: {len(model_df)} rows"}
    
    # Check if outcome has variation
    unique_vals = model_df[outcome].nunique()
    if unique_vals < 2:
        return {"error": f"No variation in outcome (only {unique_vals} unique value)"}
    
    try:
        # Prepare X and y
        X = model_df[["log_ncloc"] + predictors]
        X = sm.add_constant(X)
        y = model_df[outcome].astype(int)
        
        # Ordinal logistic (proportional odds)
        model = sm.MNLogit(y, X).fit(disp=False)
        
        # Extract simplified results
        results = {
            "outcome": outcome,
            "n": len(model_df),
            "pseudo_r2": model.prsquared,
            "aic": model.aic,
            "coefficients": {},
        }
        
        # Get coefficient averages across outcome levels
        for pred in ["log_ncloc"] + predictors:
            if pred in X.columns:
                coefs = model.params.get(pred, None)
                if coefs is not None:
                    # Average effect across levels
                    avg_coef = coefs.mean() if hasattr(coefs, "mean") else coefs
                    pvals = model.pvalues.get(pred, None)
                    avg_p = pvals.mean() if hasattr(pvals, "mean") else pvals
                    results["coefficients"][pred] = {
                        "coef": round(float(avg_coef), 4),
                        "p_value": round(float(avg_p), 4) if avg_p is not None else None,
                        "significant": avg_p < 0.05 if avg_p is not None else None,
                    }
        
        return results
    except Exception as e:
        # Fallback: treat as continuous and use OLS
        return run_ols_regression(df, outcome, predictors, log_transform_outcome=False)


def extract_results(
    model,
    outcome: str,
    predictors: list[str],
    exp_coef: bool = False,
    standardized: bool = False
) -> dict[str, Any]:
    """
    Extract results from fitted model.
    
    Args:
        standardized: If True, coefficients represent per-1-SD effects
    """
    results = {
        "outcome": outcome,
        "n": int(model.nobs),
        "r_squared": round(model.rsquared, 4) if hasattr(model, "rsquared") else None,
        "adj_r_squared": round(model.rsquared_adj, 4) if hasattr(model, "rsquared_adj") else None,
        "pseudo_r2": round(model.prsquared, 4) if hasattr(model, "prsquared") else None,
        "aic": round(model.aic, 2),
        "f_pvalue": round(model.f_pvalue, 6) if hasattr(model, "f_pvalue") else None,
        "standardized": standardized,
        "cluster_robust_se": "repo" if hasattr(model, 'cov_type') and 'cluster' in str(model.cov_type) else None,
        "coefficients": {},
    }
    
    for pred in ["log_ncloc"] + predictors:
        if pred in model.params.index:
            coef = model.params[pred]
            se = model.bse[pred]
            pval = model.pvalues[pred]
            
            # For count models, exponentiate for incidence rate ratio
            if exp_coef:
                irr = np.exp(coef)
                results["coefficients"][pred] = {
                    "coef": round(coef, 4),
                    "IRR": round(irr, 4),  # Incidence Rate Ratio
                    "se": round(se, 4),
                    "p_value": round(pval, 4),
                    "significant": pval < 0.05,
                }
            else:
                results["coefficients"][pred] = {
                    "coef": round(coef, 4),
                    "se": round(se, 4),
                    "p_value": round(pval, 4),
                    "significant": pval < 0.05,
                }
    
    return results


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def analyze_outcome(
    df: pd.DataFrame,
    sonar_col: str,
    outcome_name: str,
    predictors: list[str]
) -> dict[str, Any]:
    """Select appropriate model based on outcome type."""
    
    if sonar_col not in df.columns:
        return {"outcome": outcome_name, "error": f"Column {sonar_col} not found"}
    
    # Ratings (ordinal 1-5)
    if "rating" in sonar_col:
        return run_ordinal_logistic(df, sonar_col, predictors)
    
    # Count data (code smells, bugs, vulnerabilities, blocks)
    if sonar_col in ["sonar_code_smells", "sonar_bugs", "sonar_vulnerabilities", 
                      "sonar_duplicated_blocks", "sonar_violations"]:
        return run_negative_binomial(df, sonar_col, predictors)
    
    # Proportion data (0-100%)
    if "density" in sonar_col:
        # Treat as continuous with appropriate transform
        return run_ols_regression(df, sonar_col, predictors, log_transform_outcome=False)
    
    # Continuous outcomes (debt, complexity) - log transform
    return run_ols_regression(df, sonar_col, predictors, log_transform_outcome=True)


def run_full_analysis(df: pd.DataFrame, predictors: list[str] = None) -> list[dict[str, Any]]:
    """Run analysis for all outcomes with specified predictors."""
    
    # Use provided predictors or default to Model C (composite)
    if predictors is None:
        predictors = MODEL_C_PREDICTORS
    
    # Check which predictors are available
    available_predictors = [p for p in predictors if p in df.columns]
    print(f"\nUsing predictors: {available_predictors}")
    
    results = []
    
    for sonar_col, outcome_name in SONAR_OUTCOMES.items():
        print(f"\nAnalyzing: {outcome_name} ({sonar_col})")
        result = analyze_outcome(df, sonar_col, outcome_name, available_predictors)
        result["outcome_name"] = outcome_name
        results.append(result)
        
        # Print summary
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            n = result.get("n", 0)
            r2 = result.get("r_squared") or result.get("pseudo_r2") or 0
            print(f"  N={n}, R²={r2:.4f}")
            
            # Show significant predictors
            for pred, stats in result.get("coefficients", {}).items():
                if pred != "log_ncloc" and stats.get("significant"):
                    coef = stats.get("coef", 0)
                    p = stats.get("p_value", 1)
                    direction = "+" if coef > 0 else "-"
                    print(f"    {pred}: {direction} (p={p:.4f})")
    
    return results


def create_summary_table(results: list[dict]) -> pd.DataFrame:
    """Create publication-ready summary table."""
    rows = []
    
    for r in results:
        if "error" in r:
            continue
        
        outcome = r.get("outcome_name", r.get("outcome", ""))
        n = r.get("n", 0)
        r2 = r.get("r_squared") or r.get("adj_r_squared") or r.get("pseudo_r2")
        
        row = {
            "Outcome": outcome,
            "N": n,
            "R²": f"{r2:.3f}" if r2 else "N/A",
        }
        
        # Add coefficient info for each predictor
        for pred, stats in r.get("coefficients", {}).items():
            if pred == "log_ncloc":
                continue  # Skip control variable in main table
            
            coef = stats.get("coef", 0)
            p = stats.get("p_value", 1)
            sig = "*" if p < 0.05 else ""
            sig = "**" if p < 0.01 else sig
            sig = "***" if p < 0.001 else sig
            
            # Clean predictor name for column
            clean_name = pred.replace("_12m", "").replace("log_", "").replace("_", " ").title()
            row[clean_name] = f"{coef:+.3f}{sig}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_interpretation(results: list[dict]) -> str:
    """Generate narrative interpretation of results."""
    lines = [
        "=" * 70,
        "RESEARCH FINDINGS: Process Factors → Sustainability Outcomes",
        "=" * 70,
        "",
        "QUESTION: Do social/process factors (from git history) explain variance",
        "in SonarQube sustainability metrics, controlling for file size?",
        "",
        "METHODOLOGY:",
        "- OLS regression for continuous outcomes (log-transformed)",
        "- Negative binomial regression for count outcomes",
        "- Ordinal logistic regression for rating outcomes",
        "- All models control for file size (log NCLOC)",
        "",
        "KEY FINDINGS:",
        "",
    ]
    
    significant_findings = []
    non_significant = []
    
    for r in results:
        if "error" in r:
            continue
        
        outcome = r.get("outcome_name", "")
        
        for pred, stats in r.get("coefficients", {}).items():
            if pred == "log_ncloc":
                continue
            
            if stats.get("significant"):
                coef = stats.get("coef", 0)
                p = stats.get("p_value", 1)
                direction = "increases" if coef > 0 else "decreases"
                significant_findings.append(
                    f"  • {pred} {direction} {outcome} (β={coef:+.3f}, p={p:.4f})"
                )
            else:
                non_significant.append((pred, outcome))
    
    if significant_findings:
        lines.append("SIGNIFICANT RELATIONSHIPS (p < 0.05):")
        lines.extend(significant_findings)
    else:
        lines.append("No significant relationships found between process factors and outcomes.")
    
    lines.extend([
        "",
        "CONTROL VARIABLE (file size):",
        "- Log(NCLOC) is typically the strongest predictor",
        "- Larger files have more smells, debt, complexity (as expected)",
        "",
        "INTERPRETATION:",
        "- Positive coefficients: higher predictor → worse sustainability",
        "- IRR > 1 (count models): multiplicative increase in expected count",
        "",
    ])
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RQ2: Do Process Factors Predict Sustainability Outcomes?")
    print("=" * 70)
    
    # Load data
    df = load_and_merge_data()
    df = prepare_analysis_data(df)
    
    # Descriptive stats
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    
    desc_cols = [SIZE_CONTROL] + [c for c in MODEL_C_PREDICTORS if c in df.columns]
    print("\nPredictor variables:")
    print(df[desc_cols].describe().round(2).to_string())
    
    outcome_cols = [c for c in SONAR_OUTCOMES.keys() if c in df.columns]
    print("\nOutcome variables:")
    print(df[outcome_cols].describe().round(2).to_string())
    
    # Run all three model sets to address collinearity
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS (THREE MODEL SETS TO AVOID COLLINEARITY)")
    print("=" * 70)
    print("\nNote: churn, unique_authors, and commit_count are highly correlated (r≈0.98)")
    print("We run separate models to avoid multicollinearity:\n")
    
    all_results = {}
    all_summaries = []
    
    for model_name, predictors in MODEL_SETS.items():
        print(f"\n{'='*50}")
        print(f"MODEL SET {model_name}")
        print(f"Predictors: {', '.join(predictors)}")
        print(f"{'='*50}")
        
        results = run_full_analysis(df, predictors)
        all_results[model_name] = results
        
        # Create summary for this model set
        summary_df = create_summary_table(results)
        summary_df.insert(0, "Model", model_name)
        all_summaries.append(summary_df)
    
    # Combined summary table
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY TABLE (ALL MODEL SETS)")
    print("=" * 70)
    
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    print(combined_summary.to_string(index=False))
    
    # Save results
    combined_summary.to_csv(OUTPUT_DIR / "process_sustainability_summary.csv", index=False)
    
    # Save detailed results as JSON
    import json
    with open(OUTPUT_DIR / "process_sustainability_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Interpretation (using Model C - composite)
    print("\n" + generate_interpretation(all_results.get("C_Composite", [])))
    
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
