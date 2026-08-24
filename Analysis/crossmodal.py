# Crossmodal analýza

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLOR_INK  = '#1D1D2E'
from matplotlib.colors import LinearSegmentedColormap
CMAP_COUNTS  = LinearSegmentedColormap.from_list('counts',  ['#F5E6D3', '#E8582A', '#C4376B'])
CMAP_RESID   = LinearSegmentedColormap.from_list('resid',   ['#2B62B8', 'white',   '#C4376B'])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ACOUSTIC_CLUSTERS  = os.path.join(SCRIPT_DIR, "output_clustering_acoustic", "mci_lb_clustery.csv")
LINGUISTIC_CLUSTERS = os.path.join(SCRIPT_DIR, "output_clustering_linguistic", "mci_lb_clustery.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_crossmodal_analysis")

EXCLUDE_OUTLIERS = True

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():
    
    if not os.path.exists(ACOUSTIC_CLUSTERS):
        raise FileNotFoundError(f"Akustické clustery nenalezeny: {ACOUSTIC_CLUSTERS}")
    
    df_acoustic = pd.read_csv(ACOUSTIC_CLUSTERS)
    
    if not os.path.exists(LINGUISTIC_CLUSTERS):
        raise FileNotFoundError(f"Lingvistické clustery nenalezeny: {LINGUISTIC_CLUSTERS}")
    
    df_linguistic = pd.read_csv(LINGUISTIC_CLUSTERS)
    
    df_merged = pd.merge(
        df_acoustic[['Subject', 'cluster']],
        df_linguistic[['Subject', 'cluster']],
        on='Subject',
        how='inner',
        suffixes=('_acoustic', '_linguistic')
    )
    
    
    if EXCLUDE_OUTLIERS:
        acoustic_counts = df_merged['cluster_acoustic'].value_counts()
        linguistic_counts = df_merged['cluster_linguistic'].value_counts()
        
        outlier_acoustic = acoustic_counts[acoustic_counts == 1].index.tolist()
        outlier_linguistic = linguistic_counts[linguistic_counts == 1].index.tolist()
        
        n_before = len(df_merged)
        
        if len(outlier_acoustic) > 0:
            df_merged = df_merged[~df_merged['cluster_acoustic'].isin(outlier_acoustic)]
        
        n_before = len(df_merged)
        
        if len(outlier_linguistic) > 0:
            df_merged = df_merged[~df_merged['cluster_linguistic'].isin(outlier_linguistic)]
        
    
    return df_merged

def create_contingency_table(df):
    
    contingency = pd.crosstab(
        df['cluster_acoustic'],
        df['cluster_linguistic'],
        margins=True,
        margins_name='Total'
    )
    
    
    
    total_n = len(df)
    contingency_pct = (contingency / total_n * 100).round(1)
    
    return contingency

def chi_square_test(contingency):
    
    cont_no_margins = contingency.iloc[:-1, :-1]
    
    chi2, p_value, dof, expected = stats.chi2_contingency(cont_no_margins)
    
    
    if p_value < 0.001:
        interpretation = "PROVÁZANÉ"
    elif p_value < 0.01:
        interpretation = "PROVÁZANÉ"
    elif p_value < 0.05:
        interpretation = "PROVÁZANÉ"
    else:
        interpretation = "NEZÁVISLÉ"
    
    expected_df = pd.DataFrame(
        expected,
        index=cont_no_margins.index,
        columns=cont_no_margins.columns
    ).round(1)
    
    
    
    residuals = (cont_no_margins - expected) / np.sqrt(expected)
    residuals_df = pd.DataFrame(
        residuals,
        index=cont_no_margins.index,
        columns=cont_no_margins.columns
    ).round(2)
    
    
    return chi2, p_value, interpretation, residuals_df

def plot_heatmap(contingency, residuals, output_dir):

    cont_no_margins = contingency.iloc[:-1, :-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cont_no_margins, annot=True, fmt='d',
                cmap=CMAP_COUNTS,
                cbar_kws={'label': 'Počet pacientů'},
                linewidths=1, linecolor=COLOR_INK,
                annot_kws={'fontsize': 13, 'fontweight': 'bold', 'color': 'white'},
                ax=ax)
    ax.set_xlabel('Lingvistické shluky', fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_ylabel('Akustické shluky',    fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_title('Překrytí akustických a lingvistických subtypů\n(absolutní počty)',
                 fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.tick_params(colors=COLOR_INK)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_overlap_counts.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(residuals, annot=True, fmt='.1f',
                cmap=CMAP_RESID,
                center=0, vmin=-3, vmax=3,
                cbar_kws={'label': 'Standardizované reziduum'},
                linewidths=1, linecolor=COLOR_INK,
                annot_kws={'fontsize': 12, 'color': COLOR_INK},
                ax=ax)
    ax.set_xlabel('Lingvistické shluky', fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_ylabel('Akustické shluky',    fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_title('Standardizovaná rezidua\n(+ = více než očekáváno, - = méně než očekáváno)',
                 fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.tick_params(colors=COLOR_INK)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_overlap_residuals.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close()

def analyze_combinations(df):
    
    df['combination'] = 'A' + df['cluster_acoustic'].astype(str) + '_L' + df['cluster_linguistic'].astype(str)
    
    combo_counts = df['combination'].value_counts()
    combo_pct = (combo_counts / len(df) * 100).round(1)
    
    combo_summary = pd.DataFrame({
        'Combination': combo_counts.index,
        'N': combo_counts.values,
        'Percentage': combo_pct.values
    })
    
    for idx, row in combo_summary.iterrows():
        combo_label = row['Combination']
        n = row['N']
        pct = row['Percentage']
        
    
    
    top_combo = combo_summary.iloc[0]

    return combo_summary, df

def plot_sankey(df, combo_summary, output_dir):
    
    try:
        import plotly.graph_objects as go
        
        acoustic_clusters = sorted(df['cluster_acoustic'].unique())
        linguistic_clusters = sorted(df['cluster_linguistic'].unique())
        
        source = []
        target = []
        value = []
        
        for a_clust in acoustic_clusters:
            for l_clust in linguistic_clusters:
                count = len(df[(df['cluster_acoustic'] == a_clust) & 
                              (df['cluster_linguistic'] == l_clust)])
                if count > 0:
                    source.append(a_clust)
                    target.append(len(acoustic_clusters) + l_clust)
                    value.append(count)
        
        labels = [f'Akustický {c}' for c in acoustic_clusters] + \
                 [f'Lingvistický {c}' for c in linguistic_clusters]
        
        colors_acoustic = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6'][:len(acoustic_clusters)]
        colors_linguistic = ['#2ecc71', '#1abc9c', '#16a085', '#27ae60'][:len(linguistic_clusters)]
        node_colors = colors_acoustic + colors_linguistic
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title="Tok pacientů z akustických do lingvistických subtypů",
            font=dict(size=12),
            height=500
        )
        
        fig.write_html(os.path.join(output_dir, 'sankey_diagram.html'))
        
    except ImportError:
        pass

def concordance_analysis(df):
    
    
    acoustic_n = df['cluster_acoustic'].nunique()
    linguistic_n = df['cluster_linguistic'].nunique()
    
    if acoustic_n == linguistic_n:
        df['concordant'] = df['cluster_acoustic'] == df['cluster_linguistic']
        
        n_concordant = df['concordant'].sum()
        n_discordant = (~df['concordant']).sum()
        pct_concordant = n_concordant / len(df) * 100

    else:
        df['concordant'] = None

    return df

def main():
    ensure_dir(OUTPUT_DIR)
    
    df = load_data()
    
    contingency = create_contingency_table(df)
    
    contingency.to_csv(os.path.join(OUTPUT_DIR, 'contingency_table.csv'))
    print(f"\n Uloženo: contingency_table.csv")
    
    chi2, p_value, interpretation, residuals = chi_square_test(contingency)
    
    with open(os.path.join(OUTPUT_DIR, 'chi_square_results.txt'), 'w') as f:
        f.write("CHI-SQUARE TEST NEZÁVISLOSTI\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Chi-square: {chi2:.3f}\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Interpretace: {interpretation}\n")
    
    print(f" Uloženo: chi_square_results.txt")
    
    plot_heatmap(contingency, residuals, OUTPUT_DIR)
    
    combo_summary, df = analyze_combinations(df)
    combo_summary.to_csv(os.path.join(OUTPUT_DIR, 'dominant_combinations.csv'), index=False)
    print(f"\n Uloženo: dominant_combinations.csv")
    
    plot_sankey(df, combo_summary, OUTPUT_DIR)
    
    df = concordance_analysis(df)
    df.to_csv(os.path.join(OUTPUT_DIR, 'individual_concordance.csv'), index=False)
    print(f"\n Uloženo: individual_concordance.csv")

    print("HOTOVO!")
    print(f"Počet subjektů v analýze: {len(df)}")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
