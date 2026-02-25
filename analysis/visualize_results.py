"""
Visualization Suite for LLM-Based Holistic Software Sustainability Assessment
================================================================================

Research Focus:
- Compare LLM judgments with static-analysis tools (Sonar)
- Assess contribution of project-evolution (Git) and social signals
- Evaluate robustness and consistency of LLM-as-a-judge
- Identify potential biases in holistic sustainability assessment

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from pipeline.configs import config
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare all datasets."""
    # Load holistic assessments (main LLM output)
    holistic_df = pd.read_csv(config.holistic_assessments_path(prefer_existing=True))

    # Load source data for additional context
    sonar_df = pd.read_csv(config.sonar_metrics_path())
    git_df = pd.read_csv(config.git_metrics_path())
    
    print(f"Loaded {len(holistic_df)} holistic assessments")
    print(f"Loaded {len(sonar_df)} Sonar metrics")
    print(f"Loaded {len(git_df)} Git metrics")
    
    return holistic_df, sonar_df, git_df


# ============================================================================
# 1. LLM vs STATIC ANALYSIS CORRELATION
# ============================================================================

def plot_llm_vs_static_analysis(df, output_dir):
    """
    Research Question: How well do LLM risk judgments correlate with 
    static analysis metrics?
    
    Visualizations:
    - Correlation heatmap between LLM outputs and Sonar metrics
    - Scatter plots of risk vs key complexity metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Map risk to numeric for correlation
    risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4, 'Unknown': np.nan}
    df_analysis = df.copy()
    df_analysis['maintainability_risk_num'] = df_analysis['maintainability_risk'].map(risk_map)
    df_analysis['sustainability_risk_num'] = df_analysis['sustainability_risk'].map(risk_map)
    
    # 1a. Maintainability Risk vs Cognitive Complexity
    ax1 = axes[0, 0]
    valid_data = df_analysis.dropna(subset=['maintainability_risk_num', 'input_cognitive_complexity'])
    if len(valid_data) > 0:
        sns.boxplot(x='maintainability_risk', y='input_cognitive_complexity', 
                    data=valid_data, ax=ax1, order=['Low', 'Medium', 'High', 'Critical'])
        ax1.set_title('LLM Maintainability Risk vs Cognitive Complexity\n(Do higher complexity files get higher risk?)')
        ax1.set_xlabel('LLM Maintainability Risk')
        ax1.set_ylabel('Cognitive Complexity (Sonar)')
    
    # 1b. Sustainability Risk vs Code Smells
    ax2 = axes[0, 1]
    valid_data = df_analysis.dropna(subset=['sustainability_risk_num', 'input_code_smells'])
    if len(valid_data) > 0:
        sns.boxplot(x='sustainability_risk', y='input_code_smells',
                    data=valid_data, ax=ax2, order=['Low', 'Medium', 'High', 'Critical'])
        ax2.set_title('LLM Sustainability Risk vs Code Smells\n(Do smellier files get flagged as unsustainable?)')
        ax2.set_xlabel('LLM Sustainability Risk')
        ax2.set_ylabel('Code Smells Count (Sonar)')
    
    # 1c. Correlation heatmap
    ax3 = axes[1, 0]
    metric_cols = ['maintainability_risk_num', 'sustainability_risk_num', 'confidence',
                   'input_complexity', 'input_cognitive_complexity', 'input_code_smells',
                   'input_sqale_index', 'input_duplicated_density', 'input_ncloc']
    existing_cols = [c for c in metric_cols if c in df_analysis.columns]
    corr_matrix = df_analysis[existing_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0, ax=ax3, square=True)
    ax3.set_title('Correlation: LLM Risk vs Static Analysis Metrics')
    
    # 1d. Scatter with regression - Complexity vs Risk
    ax4 = axes[1, 1]
    valid_data = df_analysis.dropna(subset=['maintainability_risk_num', 'input_cognitive_complexity'])
    if len(valid_data) > 0:
        sns.regplot(x='input_cognitive_complexity', y='maintainability_risk_num',
                   data=valid_data, ax=ax4, scatter_kws={'alpha': 0.3})
        ax4.set_title('Cognitive Complexity → Maintainability Risk\n(Linear trend analysis)')
        ax4.set_xlabel('Cognitive Complexity')
        ax4.set_ylabel('Maintainability Risk (1=Low, 4=Critical)')
        ax4.set_yticks([1, 2, 3, 4])
        ax4.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_llm_vs_static_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 1_llm_vs_static_analysis.png")


# ============================================================================
# 2. HOLISTIC ASSESSMENT - CODE + GIT + SOCIAL
# ============================================================================

def plot_holistic_dimensions(df, output_dir):
    """
    Research Question: How do code-quality, project-evolution, and social 
    signals each contribute to sustainability assessment?
    
    Visualizations:
    - Multi-dimensional radar chart per repository
    - Flag distribution (technical vs social)
    - Risk breakdown by metric availability
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 2a. Technical vs Social Flags Distribution
    ax1 = axes[0, 0]
    flag_data = df[['n_technical_flags', 'n_social_flags']].melt(var_name='Flag Type', value_name='Count')
    flag_data['Flag Type'] = flag_data['Flag Type'].map({
        'n_technical_flags': 'Technical\n(Code Quality)',
        'n_social_flags': 'Social\n(Evolution/Collaboration)'
    })
    sns.violinplot(x='Flag Type', y='Count', data=flag_data, ax=ax1)
    ax1.set_title('Distribution of Technical vs Social Flags\n(Balance between code-quality and evolution signals)')
    ax1.set_ylabel('Number of Flags per File')
    
    # 2b. Risk by Git Data Availability (infer from input_churn_12m being null vs not)
    ax2 = axes[0, 1]
    df_git = df.copy()
    # Infer git availability from whether churn data exists
    if 'input_churn_12m' in df.columns:
        df_git['has_git_data'] = df_git['input_churn_12m'].notna().map({True: 'Git Data Available', False: 'No Git Data'})
        git_status_risk = df_git.groupby(['has_git_data', 'sustainability_risk']).size().unstack(fill_value=0)
        git_status_risk_pct = git_status_risk.div(git_status_risk.sum(axis=1), axis=0) * 100
        colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6', '#95a5a6']
        risk_order = ['Low', 'Medium', 'High', 'Critical', 'Unknown']
        existing_risks = [r for r in risk_order if r in git_status_risk_pct.columns]
        git_status_risk_pct[existing_risks].plot(kind='bar', stacked=True, ax=ax2, 
                                                  color=colors[:len(existing_risks)])
        ax2.set_title('Sustainability Risk by Git Data Availability\n(Does Git data affect risk assessment?)')
        ax2.set_xlabel('Git Data Status')
        ax2.set_ylabel('Percentage of Files')
        ax2.legend(title='Risk Level', bbox_to_anchor=(1.02, 1))
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    else:
        ax2.text(0.5, 0.5, 'No Git metrics available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Sustainability Risk by Git Data Availability')
    
    # 2c. Flags by Repository (top 8)
    ax3 = axes[1, 0]
    repo_flags = df.groupby('repo').agg({
        'n_technical_flags': 'mean',
        'n_social_flags': 'mean'
    }).sort_values('n_technical_flags', ascending=False).head(8)
    x = np.arange(len(repo_flags))
    width = 0.35
    ax3.bar(x - width/2, repo_flags['n_technical_flags'], width, label='Technical', color='#3498db')
    ax3.bar(x + width/2, repo_flags['n_social_flags'], width, label='Social', color='#e67e22')
    ax3.set_xticks(x)
    ax3.set_xticklabels(repo_flags.index, rotation=45, ha='right')
    ax3.set_title('Average Flags by Repository\n(Which projects have more technical vs social issues?)')
    ax3.set_ylabel('Average Flags per File')
    ax3.legend()
    
    # 2d. Maintainability vs Sustainability Risk Agreement
    ax4 = axes[1, 1]
    risk_agreement = pd.crosstab(df['maintainability_risk'], df['sustainability_risk'], normalize='all') * 100
    risk_order = ['Low', 'Medium', 'High', 'Critical', 'Unknown']
    existing_order_row = [r for r in risk_order if r in risk_agreement.index]
    existing_order_col = [r for r in risk_order if r in risk_agreement.columns]
    if existing_order_row and existing_order_col:
        risk_agreement = risk_agreement.reindex(index=existing_order_row, columns=existing_order_col, fill_value=0)
        sns.heatmap(risk_agreement, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Maintainability vs Sustainability Risk\n(How often do these dimensions agree?)')
        ax4.set_xlabel('Sustainability Risk')
        ax4.set_ylabel('Maintainability Risk')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_holistic_dimensions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 2_holistic_dimensions.png")


# ============================================================================
# 3. LLM ROBUSTNESS AND CONSISTENCY
# ============================================================================

def plot_robustness_analysis(df, output_dir):
    """
    Research Question: How robust and consistent are LLM judgments?
    
    Visualizations:
    - Agreement rate distribution
    - Confidence vs Agreement correlation
    - Latency vs Response quality
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 3a. LLM Confidence Distribution by Risk Level
    ax1 = axes[0, 0]
    risk_order = ['Low', 'Medium', 'High', 'Critical']
    valid_risks = [r for r in risk_order if r in df['maintainability_risk'].values]
    if valid_risks:
        sns.boxplot(x='maintainability_risk', y='confidence', data=df, ax=ax1, order=valid_risks)
        ax1.set_title('LLM Confidence by Risk Level\n(Is the model more confident about certain risk levels?)')
        ax1.set_xlabel('Maintainability Risk')
        ax1.set_ylabel('Confidence Score')
    
    # 3b. Response Latency Distribution
    ax2 = axes[0, 1]
    if 'latency_ms' in df.columns:
        df['latency_sec'] = df['latency_ms'] / 1000
        sns.histplot(df['latency_sec'], bins=50, ax=ax2, kde=True)
        ax2.axvline(df['latency_sec'].median(), color='r', linestyle='--', label=f'Median: {df["latency_sec"].median():.1f}s')
        ax2.set_title('Response Latency Distribution\n(Inference time consistency)')
        ax2.set_xlabel('Latency (seconds)')
        ax2.set_ylabel('Count')
        ax2.legend()
    
    # 3c. Confidence vs File Complexity
    ax3 = axes[1, 0]
    valid_data = df.dropna(subset=['confidence', 'input_cognitive_complexity'])
    if len(valid_data) > 0:
        sns.scatterplot(x='input_cognitive_complexity', y='confidence', 
                        hue='maintainability_risk', data=valid_data, ax=ax3, alpha=0.5)
        ax3.set_title('LLM Confidence vs File Complexity\n(Does complexity affect model certainty?)')
        ax3.set_xlabel('Cognitive Complexity')
        ax3.set_ylabel('Confidence')
        ax3.legend(title='Risk', bbox_to_anchor=(1.02, 1))
    
    # 3d. Risk Distribution Consistency across Repos
    ax4 = axes[1, 1]
    risk_by_repo = pd.crosstab(df['repo'], df['maintainability_risk'], normalize='index') * 100
    risk_order = ['Low', 'Medium', 'High', 'Critical', 'Unknown']
    existing_risks = [r for r in risk_order if r in risk_by_repo.columns]
    if existing_risks:
        risk_by_repo = risk_by_repo[existing_risks]
        risk_by_repo.plot(kind='barh', stacked=True, ax=ax4, 
                          color=['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6', '#95a5a6'][:len(existing_risks)])
        ax4.set_title('Risk Distribution by Repository\n(Is risk assessment consistent across projects?)')
        ax4.set_xlabel('Percentage of Files')
        ax4.set_ylabel('Repository')
        ax4.legend(title='Risk', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_robustness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 3_robustness_analysis.png")


# ============================================================================
# 4. BIAS AND FAIRNESS ANALYSIS
# ============================================================================

def plot_bias_analysis(df, output_dir):
    """
    Research Question: Are there biases in LLM sustainability judgments?
    
    Visualizations:
    - Risk distribution by programming language
    - Risk vs file size (potential size bias)
    - Flag consistency validation
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 4a. Risk Distribution by Language
    ax1 = axes[0, 0]
    lang_counts = df['language'].value_counts()
    top_langs = lang_counts[lang_counts >= 10].index.tolist()[:8]
    df_top_langs = df[df['language'].isin(top_langs)]
    risk_by_lang = pd.crosstab(df_top_langs['language'], df_top_langs['maintainability_risk'], 
                                normalize='index') * 100
    risk_order = ['Low', 'Medium', 'High', 'Critical', 'Unknown']
    existing_risks = [r for r in risk_order if r in risk_by_lang.columns]
    if existing_risks:
        risk_by_lang[existing_risks].plot(kind='bar', stacked=True, ax=ax1,
                                           color=['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6', '#95a5a6'][:len(existing_risks)])
        ax1.set_title('Maintainability Risk by Language\n(Is there language bias in risk assessment?)')
        ax1.set_xlabel('Programming Language')
        ax1.set_ylabel('Percentage')
        ax1.legend(title='Risk', bbox_to_anchor=(1.02, 1))
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 4b. Risk vs File Size (NCLOC)
    ax2 = axes[0, 1]
    risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df_analysis = df.copy()
    df_analysis['risk_num'] = df_analysis['maintainability_risk'].map(risk_map)
    valid_data = df_analysis.dropna(subset=['risk_num', 'input_ncloc'])
    if len(valid_data) > 0:
        # Bin file sizes
        valid_data['size_bin'] = pd.cut(valid_data['input_ncloc'], 
                                         bins=[0, 50, 100, 200, 500, float('inf')],
                                         labels=['≤50', '51-100', '101-200', '201-500', '>500'])
        size_risk = valid_data.groupby('size_bin')['risk_num'].mean()
        size_risk.plot(kind='bar', ax=ax2, color='#3498db')
        ax2.set_title('Average Risk by File Size\n(Size bias detection)')
        ax2.set_xlabel('Lines of Code')
        ax2.set_ylabel('Average Risk (1=Low, 4=Critical)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.axhline(valid_data['risk_num'].mean(), color='r', linestyle='--', 
                    label=f'Overall mean: {valid_data["risk_num"].mean():.2f}')
        ax2.legend()
    
    # 4c. Deterministic Flag Agreement
    ax3 = axes[1, 0]
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    if flag_cols:
        flag_sums = df[flag_cols].sum().sort_values(ascending=True)
        flag_sums.plot(kind='barh', ax=ax3, color='#9b59b6')
        ax3.set_title('Deterministic Flag Frequency\n(Threshold-based flags for validation)')
        ax3.set_xlabel('Number of Files Flagged')
        ax3.set_ylabel('')
    
    # 4d. LLM Violations by Repository (if available)
    ax4 = axes[1, 1]
    if 'n_llm_violations' in df.columns:
        violations_by_repo = df.groupby('repo')['n_llm_violations'].agg(['sum', 'mean'])
        violations_by_repo = violations_by_repo.sort_values('sum', ascending=True)
        violations_by_repo['sum'].plot(kind='barh', ax=ax4, color='#e74c3c')
        ax4.set_title('LLM Flag Violations by Repository\n(Where does LLM contradict thresholds?)')
        ax4.set_xlabel('Total Violations')
        ax4.set_ylabel('Repository')
    else:
        # Show confidence distribution instead
        sns.histplot(df['confidence'], bins=20, ax=ax4, kde=True)
        ax4.set_title('LLM Confidence Distribution\n(Model certainty across assessments)')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_bias_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 4_bias_analysis.png")


# ============================================================================
# 5. MULTI-SIGNAL INTEGRATION VALUE
# ============================================================================

def plot_signal_integration(df, git_df, output_dir):
    """
    Research Question: What is the value of combining code, git, and social 
    signals vs using static analysis alone?
    
    Visualizations:
    - Risk changes with Git data availability
    - Social signals impact on sustainability
    - Driver analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 5a. Top Drivers of Risk (word frequency)
    ax1 = axes[0, 0]
    if 'top_drivers' in df.columns:
        all_drivers = df['top_drivers'].dropna().str.split('; ').explode()
        driver_counts = all_drivers.value_counts().head(15)
        driver_counts.plot(kind='barh', ax=ax1, color='#16a085')
        ax1.set_title('Most Frequent Risk Drivers\n(What signals drive risk assessment?)')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Driver')
    
    # 5b. Git Signal Contribution
    ax2 = axes[0, 1]
    git_signals = ['input_churn_12m', 'input_unique_authors', 'input_recency_days']
    existing_signals = [s for s in git_signals if s in df.columns]
    if existing_signals:
        # Show correlation between git signals and sustainability risk
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df_analysis = df.copy()
        df_analysis['sust_risk_num'] = df_analysis['sustainability_risk'].map(risk_map)
        correlations = df_analysis[existing_signals + ['sust_risk_num']].corr()['sust_risk_num'].drop('sust_risk_num')
        correlations.plot(kind='bar', ax=ax2, color=['#e74c3c' if v < 0 else '#2ecc71' for v in correlations])
        ax2.set_title('Git Signals Correlation with Sustainability Risk\n(Which evolution metrics matter?)')
        ax2.set_xlabel('Git Metric')
        ax2.set_ylabel('Correlation with Sustainability Risk')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 5c. Social Signal Impact
    ax3 = axes[1, 0]
    social_cols = ['input_unique_authors', 'input_dominant_author_share', 'input_single_contributor']
    existing_social = [s for s in social_cols if s in df.columns]
    if existing_social and 'sustainability_risk' in df.columns:
        # Show bus factor risk across repos
        df_with_git = df[df['git_metrics_status'] == 'ok'] if 'git_metrics_status' in df.columns else df
        if 'input_unique_authors' in df_with_git.columns:
            auth_risk = df_with_git.groupby('sustainability_risk')['input_unique_authors'].mean()
            risk_order = ['Low', 'Medium', 'High', 'Critical']
            existing_risks = [r for r in risk_order if r in auth_risk.index]
            if existing_risks:
                auth_risk.reindex(existing_risks).plot(kind='bar', ax=ax3, color='#8e44ad')
                ax3.set_title('Average Contributors by Sustainability Risk\n(Do fewer contributors = higher risk?)')
                ax3.set_xlabel('Sustainability Risk')
                ax3.set_ylabel('Average Unique Authors (12m)')
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 5d. Evaluation Status Distribution (or Assessment Summary if not available)
    ax4 = axes[1, 1]
    if 'evaluation_status' in df.columns:
        status_counts = df['evaluation_status'].value_counts()
        colors = ['#2ecc71' if s == 'success' else '#e74c3c' if s == 'error' else '#f1c40f' 
                  for s in status_counts.index]
        status_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Evaluation Status Distribution\n(Assessment success rate)')
        ax4.set_ylabel('')
    else:
        # Show risk agreement instead
        risk_counts = df['maintainability_risk'].value_counts()
        colors = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c', 'Critical': '#9b59b6', 'Unknown': '#95a5a6'}
        risk_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%',
                        colors=[colors.get(r, '#95a5a6') for r in risk_counts.index])
        ax4.set_title('Maintainability Risk Distribution\n(Overall assessment breakdown)')
        ax4.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_signal_integration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 5_signal_integration.png")


# ============================================================================
# 6. SUMMARY DASHBOARD
# ============================================================================

def plot_summary_dashboard(df, output_dir):
    """
    Executive summary of the holistic sustainability assessment.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 6a. Overall Risk Distribution (Maintainability)
    ax1 = fig.add_subplot(gs[0, 0])
    maint_risk = df['maintainability_risk'].value_counts()
    colors = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c', 'Critical': '#9b59b6', 'Unknown': '#95a5a6'}
    maint_risk.plot(kind='pie', ax=ax1, autopct='%1.1f%%', 
                    colors=[colors.get(r, '#95a5a6') for r in maint_risk.index])
    ax1.set_title('Maintainability Risk\nDistribution')
    ax1.set_ylabel('')
    
    # 6b. Overall Risk Distribution (Sustainability)
    ax2 = fig.add_subplot(gs[0, 1])
    sust_risk = df['sustainability_risk'].value_counts()
    sust_risk.plot(kind='pie', ax=ax2, autopct='%1.1f%%',
                   colors=[colors.get(r, '#95a5a6') for r in sust_risk.index])
    ax2.set_title('Sustainability Risk\nDistribution')
    ax2.set_ylabel('')
    
    # 6c. Key Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = f"""
    ASSESSMENT SUMMARY
    ==================
    
    Total Files Assessed: {len(df):,}
    Unique Repositories: {df['repo'].nunique()}
    Languages Covered: {df['language'].nunique()}
    
    Model: {df['model_name'].iloc[0] if 'model_name' in df.columns else 'N/A'}
    Temperature: {df['temperature'].iloc[0] if 'temperature' in df.columns else 'N/A'}
    
    Git Data Available: {(df['input_churn_12m'].notna()).sum() if 'input_churn_12m' in df.columns else 'N/A'}
    Assessment Count: {len(df):,} 
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes)
    
    # 6d. Risk by Repository
    ax4 = fig.add_subplot(gs[1, :])
    risk_by_repo = pd.crosstab(df['repo'], df['maintainability_risk'], normalize='index') * 100
    risk_order = ['Low', 'Medium', 'High', 'Critical', 'Unknown']
    existing_risks = [r for r in risk_order if r in risk_by_repo.columns]
    if existing_risks:
        risk_by_repo = risk_by_repo[existing_risks]
        risk_by_repo.plot(kind='bar', stacked=True, ax=ax4,
                          color=[colors.get(r, '#95a5a6') for r in existing_risks])
        ax4.set_title('Maintainability Risk Distribution by Repository')
        ax4.set_xlabel('Repository')
        ax4.set_ylabel('Percentage of Files')
        ax4.legend(title='Risk', bbox_to_anchor=(1.02, 1))
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 6e. Metrics Correlation Summary
    ax5 = fig.add_subplot(gs[2, :2])
    risk_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df_corr = df.copy()
    df_corr['maint_risk_num'] = df_corr['maintainability_risk'].map(risk_map)
    df_corr['sust_risk_num'] = df_corr['sustainability_risk'].map(risk_map)
    
    key_metrics = ['input_cognitive_complexity', 'input_code_smells', 'input_sqale_index',
                   'input_churn_12m', 'input_unique_authors', 'confidence']
    existing_metrics = [m for m in key_metrics if m in df_corr.columns]
    if existing_metrics:
        corr_with_risk = df_corr[existing_metrics + ['maint_risk_num', 'sust_risk_num']].corr()[['maint_risk_num', 'sust_risk_num']]
        corr_with_risk = corr_with_risk.drop(['maint_risk_num', 'sust_risk_num'])
        corr_with_risk.columns = ['Maintainability', 'Sustainability']
        corr_with_risk.plot(kind='barh', ax=ax5)
        ax5.set_title('Metric Correlations with Risk Levels')
        ax5.set_xlabel('Correlation Coefficient')
        ax5.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax5.legend(title='Risk Type')
    
    # 6f. Flag Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    if flag_cols:
        top_flags = df[flag_cols].sum().sort_values(ascending=False).head(8)
        flags_text = "TOP FLAGS\n" + "=" * 20 + "\n\n"
        for flag, count in top_flags.items():
            flags_text += f"{flag[5:]}: {count:,}\n"
        ax6.text(0.1, 0.5, flags_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax6.transAxes)
    
    plt.savefig(output_dir / '6_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 6_summary_dashboard.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("HOLISTIC SUSTAINABILITY ASSESSMENT - VISUALIZATION SUITE")
    print("=" * 70)
    print()
    
    # Load data
    holistic_df, sonar_df, git_df = load_data()
    
    # Create output directory
    output_dir = Path("data/results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 40)
    
    plot_llm_vs_static_analysis(holistic_df, output_dir)
    plot_holistic_dimensions(holistic_df, output_dir)
    plot_robustness_analysis(holistic_df, output_dir)
    plot_bias_analysis(holistic_df, output_dir)
    plot_signal_integration(holistic_df, git_df, output_dir)
    plot_summary_dashboard(holistic_df, output_dir)
    
    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated 6 visualization sets in: {output_dir.absolute()}")
    print("""
Visualizations created:
1. LLM vs Static Analysis - Correlation and agreement analysis
2. Holistic Dimensions - Technical vs Social signal breakdown
3. Robustness Analysis - LLM confidence and consistency
4. Bias Analysis - Language and size bias detection
5. Signal Integration - Value of multi-signal approach
6. Summary Dashboard - Executive overview

These visualizations address the research gaps:
• Comparing LLM judgment with static-analysis tools
• Holistic assessment combining code-quality + evolution + social signals
• Robustness considerations from LLM-as-a-judge literature
    """)


if __name__ == "__main__":
    main()
