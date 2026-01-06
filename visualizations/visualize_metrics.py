"""
Git Metrics Visualization Script
Provides interactive tables and charts for analyzing repository health metrics,
including data quality summaries for reporting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = Path("data/results/git_metrics.csv")
OUTPUT_DIR = Path("data/results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOC_BANDS = [("50-150", 50, 150), ("151-275", 151, 275), ("276-400", 276, 400)]
CHURN_BINS = [-1, 0, 50, 200, 500, 1000, float("inf")]
CHURN_LABELS = ["0", "1-50", "51-200", "201-500", "501-1000", "1001+"]


def load_data():
    """Load and prepare the git metrics data."""
    df = pd.read_csv(DATA_PATH)
    # Clean up boolean columns
    df['bus_factor_single_dev'] = df['bus_factor_single_dev'].map({'True': True, 'False': False, True: True, False: False})
    df['bus_factor_75_dominant_author'] = df['bus_factor_75_dominant_author'].map({'True': True, 'False': False, True: True, False: False})
    numeric_cols = [
        'lines_of_code',
        'unique_authors_12m',
        'repo_unique_authors_12m',
        'repo_commits_12m',
        'repo_commits_per_month_12m',
        'churn_12m',
        'added_lines_12m',
        'deleted_lines_12m',
        'dominant_author_share',
        'last_12m_observed_commits',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def repo_summary_table(df):
    """Create a summary table grouped by repository."""
    summary = df.groupby(['repo', 'category']).agg({
        'file_path': 'count',
        'lines_of_code': 'sum',
        'unique_authors_12m': 'mean',
        'churn_12m': 'sum',
        'added_lines_12m': 'sum',
        'deleted_lines_12m': 'sum',
        'bus_factor_single_dev': 'sum',
        'dominant_author_share': 'mean',
        'repo_commits_12m': 'first',
        'repo_unique_authors_12m': 'first'
    }).reset_index()
    
    summary.columns = [
        'Repository', 'Category', 'Total Files', 'Total LOC', 
        'Avg Unique Authors', 'Total Churn', 'Lines Added', 'Lines Deleted',
        'Single Dev Files', 'Avg Dominant Share', 'Repo Commits (12m)', 'Repo Authors (12m)'
    ]
    
    # Round numeric columns
    summary['Avg Unique Authors'] = summary['Avg Unique Authors'].round(2)
    summary['Avg Dominant Share'] = summary['Avg Dominant Share'].round(2)
    
    return summary


def risk_analysis_table(df):
    """Identify high-risk files based on bus factor and concentration."""
    risk_df = df[
        (df['bus_factor_single_dev'] == True) | 
        (df['dominant_author_share'] >= 0.75)
    ].copy()
    
    risk_df['risk_score'] = (
        risk_df['bus_factor_single_dev'].astype(int) * 2 +
        (risk_df['dominant_author_share'] >= 0.9).astype(int) * 2 +
        (risk_df['dominant_author_share'] >= 0.75).astype(int)
    )
    
    risk_df = risk_df.sort_values('risk_score', ascending=False)
    
    return risk_df[['repo', 'file_path', 'lines_of_code', 'unique_authors_12m', 
                    'dominant_author', 'dominant_author_share', 'bus_factor_single_dev', 
                    'churn_12m', 'risk_score']]


def _assign_loc_band(loc_value):
    if pd.isna(loc_value):
        return "unknown"
    for label, lower, upper in LOC_BANDS:
        if lower <= loc_value <= upper:
            return label
    return "unknown"


def data_quality_summary(df):
    """Compute dataset-level validation stats for reporting."""
    loc = df['lines_of_code']
    churn = df['churn_12m']
    dominant_share = df['dominant_author_share'].fillna(0)
    single_dev = df['bus_factor_single_dev'].fillna(False)

    summary = {
        "total_files": len(df),
        "total_repos": df['repo'].nunique(),
        "loc_min": loc.min(),
        "loc_median": loc.median(),
        "loc_mean": loc.mean(),
        "loc_max": loc.max(),
        "churn_min": churn.min(),
        "churn_median": churn.median(),
        "churn_mean": churn.mean(),
        "churn_max": churn.max(),
        "single_dev_pct": single_dev.mean() * 100,
        "dominant_share_75_pct": (dominant_share >= 0.75).mean() * 100,
        "dominant_share_90_pct": (dominant_share >= 0.90).mean() * 100,
    }

    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    return summary_df


def loc_band_distribution(df):
    """Count representative files by LOC band."""
    loc_df = df.copy()
    loc_df['loc_band'] = loc_df['lines_of_code'].apply(_assign_loc_band)
    loc_df['loc_band'] = pd.Categorical(
        loc_df['loc_band'],
        categories=[band[0] for band in LOC_BANDS] + ["unknown"],
        ordered=True,
    )
    counts = loc_df['loc_band'].value_counts(dropna=False).sort_index()
    total = counts.sum() or 1
    return pd.DataFrame(
        {
            "loc_band": counts.index.astype(str),
            "file_count": counts.values,
            "percent_files": (counts.values / total) * 100,
        }
    )


def churn_band_distribution(df):
    """Bucket churn into bands for reporting."""
    churn = df['churn_12m'].fillna(0)
    churn_band = pd.cut(churn, bins=CHURN_BINS, labels=CHURN_LABELS)
    counts = churn_band.value_counts(dropna=False).reindex(CHURN_LABELS, fill_value=0)
    total = counts.sum() or 1
    return pd.DataFrame(
        {
            "churn_band": counts.index.astype(str),
            "file_count": counts.values,
            "percent_files": (counts.values / total) * 100,
        }
    )




def plot_repo_overview(df):
    """Create overview charts for repositories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Repository Health Overview', fontsize=14, fontweight='bold')
    
    # 1. Files per repo
    repo_files = df.groupby('repo')['file_path'].count().sort_values(ascending=True)
    axes[0, 0].barh(repo_files.index, repo_files.values, color='steelblue')
    axes[0, 0].set_xlabel('Number of Files')
    axes[0, 0].set_title('Files per Repository')
    
    # 2. Category distribution
    category_counts = df['category'].value_counts()
    colors = {'STAGNANT': '#e74c3c', 'ACTIVE': '#27ae60', 'MODERATE': '#f39c12'}
    cat_colors = [colors.get(c, '#3498db') for c in category_counts.index]
    axes[0, 1].pie(category_counts.values, labels=category_counts.index, 
                   autopct='%1.1f%%', colors=cat_colors)
    axes[0, 1].set_title('Files by Category')
    
    # 3. Bus factor risk by repo
    bus_factor_risk = df.groupby('repo')['bus_factor_single_dev'].mean() * 100
    bus_factor_risk = bus_factor_risk.sort_values(ascending=True)
    colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 25 else '#27ae60' for v in bus_factor_risk.values]
    axes[1, 0].barh(bus_factor_risk.index, bus_factor_risk.values, color=colors)
    axes[1, 0].set_xlabel('% Files with Single Developer')
    axes[1, 0].set_title('Bus Factor Risk by Repository')
    axes[1, 0].axvline(x=50, color='red', linestyle='--', alpha=0.5)
    
    # 4. Churn distribution
    churn_by_repo = df.groupby('repo')['churn_12m'].sum().sort_values(ascending=True)
    axes[1, 1].barh(churn_by_repo.index, churn_by_repo.values, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Total Churn (Lines Changed)')
    axes[1, 1].set_title('Code Churn by Repository (12 months)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'repo_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_DIR / 'repo_overview.png'}")


def plot_author_concentration(df):
    """Visualize author concentration/knowledge silos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Author Concentration Analysis', fontsize=14, fontweight='bold')
    
    # 1. Dominant author share distribution
    axes[0].hist(df['dominant_author_share'].dropna(), bins=20, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0.75, color='red', linestyle='--', label='75% threshold')
    axes[0].axvline(x=0.5, color='orange', linestyle='--', label='50% threshold')
    axes[0].set_xlabel('Dominant Author Share')
    axes[0].set_ylabel('Number of Files')
    axes[0].set_title('Distribution of Author Concentration')
    axes[0].legend()
    
    # 2. Top contributors across all repos
    top_authors = df[df['dominant_author'] != ''].groupby('dominant_author').agg({
        'file_path': 'count',
        'lines_of_code': 'sum'
    }).sort_values('file_path', ascending=False).head(15)
    
    axes[1].barh(top_authors.index, top_authors['file_path'], color='teal')
    axes[1].set_xlabel('Number of Files Dominated')
    axes[1].set_title('Top 15 Dominant Authors')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'author_concentration.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_DIR / 'author_concentration.png'}")


def plot_heatmap(df):
    """Create heatmap of key metrics by repository."""
    # Aggregate metrics by repo
    heatmap_data = df.groupby('repo').agg({
        'lines_of_code': 'sum',
        'unique_authors_12m': 'mean',
        'churn_12m': 'sum',
        'dominant_author_share': 'mean',
        'bus_factor_single_dev': 'mean'
    })
    
    # Normalize for heatmap
    heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    heatmap_normalized.columns = ['LOC', 'Avg Authors', 'Churn', 'Dominant Share', 'Bus Factor Risk']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                linewidths=0.5, cbar_kws={'label': 'Normalized Value (0-1)'})
    plt.title('Repository Health Metrics Heatmap\n(Higher = More Risk for Dominant Share & Bus Factor)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_DIR / 'metrics_heatmap.png'}")


def interactive_display(df):
    """Display interactive summary tables."""
    print("\n" + "="*80)
    print("GIT METRICS ANALYSIS REPORT")
    print("="*80)
    
    # Overall stats
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Files Analyzed: {len(df):,}")
    print(f"   Total Repositories: {df['repo'].nunique()}")
    print(f"   Total Lines of Code: {df['lines_of_code'].sum():,}")
    print(f"   Files with Single Developer: {df['bus_factor_single_dev'].sum():,} ({df['bus_factor_single_dev'].mean()*100:.1f}%)")

    quality_summary = data_quality_summary(df)
    loc_summary = loc_band_distribution(df)
    churn_summary = churn_band_distribution(df)

    print("\nDATA QUALITY SUMMARY")
    for _, row in quality_summary.iterrows():
        if isinstance(row["value"], float):
            value = f"{row['value']:.2f}"
        else:
            value = row["value"]
        print(f"   {row['metric']}: {value}")

    print("\nLOC BAND DISTRIBUTION")
    for _, row in loc_summary.iterrows():
        print(f"   {row['loc_band']}: {int(row['file_count'])} files ({row['percent_files']:.1f}%)")

    print("\nCHURN BAND DISTRIBUTION")
    for _, row in churn_summary.iterrows():
        print(f"   {row['churn_band']}: {int(row['file_count'])} files ({row['percent_files']:.1f}%)")
    
    # Category breakdown
    print(f"\n📁 CATEGORY BREAKDOWN")
    for cat, count in df['category'].value_counts().items():
        print(f"   {cat}: {count:,} files ({count/len(df)*100:.1f}%)")
    
    # Repository summary
    print(f"\n📈 REPOSITORY SUMMARY")
    summary = repo_summary_table(df)
    print(summary.to_string(index=False))
    
    # High risk files
    print(f"\n⚠️  HIGH RISK FILES (Top 20)")
    risk_df = risk_analysis_table(df).head(20)
    if len(risk_df) > 0:
        print(risk_df.to_string(index=False))
    else:
        print("   No high-risk files identified.")
    
    # Save tables to CSV
    summary.to_csv(OUTPUT_DIR / 'repo_summary.csv', index=False)
    risk_analysis_table(df).to_csv(OUTPUT_DIR / 'risk_analysis.csv', index=False)
    quality_summary.to_csv(OUTPUT_DIR / 'data_quality_summary.csv', index=False)
    loc_summary.to_csv(OUTPUT_DIR / 'loc_band_distribution.csv', index=False)
    churn_summary.to_csv(OUTPUT_DIR / 'churn_distribution.csv', index=False)
    print(f"\n💾 Tables saved to {OUTPUT_DIR}/")


def main():
    """Main entry point."""
    print("Loading git metrics data...")
    df = load_data()
    print(f"Loaded {len(df):,} records from {df['repo'].nunique()} repositories")
    
    # Display interactive summary
    interactive_display(df)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_repo_overview(df)
    plot_author_concentration(df)
    plot_heatmap(df)
    
    print("\n✅ Analysis complete!")
    print(f"   Visualizations saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
