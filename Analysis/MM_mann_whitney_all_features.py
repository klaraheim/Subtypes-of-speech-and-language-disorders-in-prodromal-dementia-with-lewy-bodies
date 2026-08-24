# MANN-WHITNEY U test + boxploty (multimodální, kombinace akustika + lingvistika, všechny parametry individuálně)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

GROUP_COLORS = {
    0:    '#E8582A',
    1:    '#C4376B',
    2:    '#7B3FA0',
    3:    '#2B62B8',
    'HC': '#3D5A80',
}
MEDIAN_COLOR   = '#1D1D2E'
MEAN_COLOR     = '#E8582A'
SIG_LINE_COLOR = '#1D1D2E'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_FILE = os.path.join(SCRIPT_DIR, "output_clustering_multimodal", "mci_lb_clustery.csv")
ACOUSTIC_DATA = os.path.join(SCRIPT_DIR, "acoustic_residualized_filtered.csv")
LINGUISTIC_DATA = os.path.join(SCRIPT_DIR, "linguistic_residualized_filtered.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_mann_whitney_multimodal_all_features")

SUBJECT_COL = "Subject"
GROUP_COL = "Category"
HC_LABEL = "HC"
TARGET_LABEL = "MCI-LB"

ALPHA = 0.05

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():
    
    if not os.path.exists(CLUSTER_FILE):
        raise FileNotFoundError(f"Soubor nenalezen: {CLUSTER_FILE}")
    
    df_clusters = pd.read_csv(CLUSTER_FILE)
    
    if not os.path.exists(ACOUSTIC_DATA):
        raise FileNotFoundError(f"Soubor nenalezen: {ACOUSTIC_DATA}")
    
    df_acoustic = pd.read_csv(ACOUSTIC_DATA)
    
    exclude_cols = [SUBJECT_COL, GROUP_COL, 'MOCA', 'education type', 
                    'memory z-score', 'visuo-spatial z-score', 
                    'attention z-score', 'executive function z-score', 'GDS']
    acoustic_features = [c for c in df_acoustic.columns if c not in exclude_cols]
    
    if not os.path.exists(LINGUISTIC_DATA):
        raise FileNotFoundError(f"Soubor nenalezen: {LINGUISTIC_DATA}")
    
    df_linguistic = pd.read_csv(LINGUISTIC_DATA)
    
    linguistic_features = [c for c in df_linguistic.columns 
                          if c not in [SUBJECT_COL, GROUP_COL]]
    
    df_acoustic_subset = df_acoustic[[SUBJECT_COL, GROUP_COL] + acoustic_features]
    df_linguistic_subset = df_linguistic[[SUBJECT_COL, GROUP_COL] + linguistic_features]
    
    df_merged = pd.merge(df_acoustic_subset, df_linguistic_subset, 
                         on=[SUBJECT_COL, GROUP_COL], how='inner')
    
    
    df_mcilb = df_merged[df_merged[GROUP_COL] == TARGET_LABEL].copy()
    df_mcilb_with_clusters = pd.merge(
        df_mcilb,
        df_clusters[[SUBJECT_COL, 'cluster']],
        on=SUBJECT_COL,
        how='inner'
    )
    
    df_hc = df_merged[df_merged[GROUP_COL] == HC_LABEL].copy()
    df_hc['cluster'] = 'HC'
    
    df_all = pd.concat([df_mcilb_with_clusters, df_hc], ignore_index=True)
    
    all_features = acoustic_features + linguistic_features
    
    
    return df_all, all_features, acoustic_features, linguistic_features

def mann_whitney_tests(df, feature_col):
    unique_groups = df['cluster'].unique()
    numeric_groups = sorted([g for g in unique_groups if g != 'HC'])
    str_groups = [g for g in unique_groups if g == 'HC']
    groups = numeric_groups + str_groups
    
    results = []
    
    for g1, g2 in combinations(groups, 2):
        data1 = df[df['cluster'] == g1][feature_col].dropna()
        data2 = df[df['cluster'] == g2][feature_col].dropna()
        
        if len(data1) < 3 or len(data2) < 3:
            continue
        
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        
        results.append({
            'Group1': g1,
            'Group2': g2,
            'n1': len(data1),
            'n2': len(data2),
            'median1': data1.median(),
            'median2': data2.median(),
            'U_statistic': stat,
            'p_value': p_value
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        return df_results
    
    _, p_fdr, _, _ = multipletests(df_results['p_value'], alpha=ALPHA, method='fdr_bh')
    df_results['p_fdr'] = p_fdr
    df_results['significant'] = df_results['p_fdr'] < ALPHA
    
    return df_results

def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def plot_boxplot_with_mw(df, feature_col, mw_results, output_dir, is_acoustic=True):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 8))

    groups = sorted([g for g in df['cluster'].unique() if g != 'HC']) + ['HC']
    data_groups = []
    labels = []
    colors = []

    for g in groups:
        data = df[df['cluster'] == g][feature_col].dropna()
        data_groups.append(data)
        colors.append(GROUP_COLORS.get(g, '#4A90D4'))
        if g == 'HC':
            labels.append(f'HC\n(n={len(data)})')
        else:
            labels.append(f'Shluk {g}\n(n={len(data)})')

    bp = ax.boxplot(
        data_groups,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5),
        meanprops=dict(color=MEAN_COLOR, linewidth=2.5, linestyle='--'),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=5, alpha=0.5),
        boxprops=dict(linewidth=1.5)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    for i, (w0, w1, c0, c1) in enumerate(zip(
        bp['whiskers'][::2], bp['whiskers'][1::2],
        bp['caps'][::2],     bp['caps'][1::2]
    )):
        c = colors[i]
        w0.set_color(c); w1.set_color(c)
        c0.set_color(c); c1.set_color(c)

    for i, flier in enumerate(bp['fliers']):
        flier.set_markerfacecolor(colors[i])
        flier.set_markeredgecolor(colors[i])

    if len(data_groups) > 0 and len(mw_results) > 0:
        y_max = max([d.max() for d in data_groups if len(d) > 0])
        y_min = min([d.min() for d in data_groups if len(d) > 0])
        y_range = y_max - y_min
        offset = 0.05 * y_range
        current_y = y_max + offset

        sig_rows = mw_results[mw_results['significant']]
        n_sig = len(sig_rows)

        for _, row in sig_rows.iterrows():
            g1 = row['Group1']
            g2 = row['Group2']
            try:
                idx1 = groups.index(g1) + 1
                idx2 = groups.index(g2) + 1
            except ValueError:
                continue

            stars = get_significance_stars(row['p_fdr'])
            ax.plot([idx1, idx2], [current_y, current_y],
                    color=SIG_LINE_COLOR, linewidth=1.5)
            ax.text((idx1 + idx2) / 2, current_y + offset * 0.3, stars,
                    ha='center', va='bottom', fontsize=16,
                    fontweight='bold', color=SIG_LINE_COLOR)
            current_y += offset * 1.5

        y_top = current_y + offset * 1.5 if n_sig > 0 else y_max + offset * 2
        ax.set_ylim(bottom=y_min - offset, top=y_top)

    modality = "Akustika" if is_acoustic else "Lingvistika"
    ax.set_ylabel('Hodnota parametru', fontsize=18, fontweight='bold', color=MEDIAN_COLOR)
    ax.set_title(f'{feature_col}\n[{modality}]', fontsize=18, fontweight='bold', color=MEDIAN_COLOR)
    ax.grid(axis='y', alpha=0.25, linestyle='--', color=MEDIAN_COLOR)
    ax.tick_params(colors=MEDIAN_COLOR, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(MEDIAN_COLOR)
        spine.set_linewidth(0.8)

    legend_elements = [
        Line2D([0], [0], color=MEDIAN_COLOR, linewidth=2.5, label='Medián'),
        Line2D([0], [0], color=MEAN_COLOR,   linewidth=2.5, linestyle='--', label='Průměr'),
    ]
    for g, color in zip(groups, colors):
        label = 'HC' if g == 'HC' else f'Shluk {g}'
        legend_elements.append(
            mpatches.Patch(facecolor=color, alpha=0.55, edgecolor=color, label=label)
        )
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=16)

    plt.tight_layout()

    safe_name = feature_col.replace(' ', '_').replace('/', '_').replace('-', '_')
    plt.savefig(os.path.join(output_dir, f'boxplot_{safe_name}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'boxplot_{safe_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    df_all, all_features, acoustic_features, linguistic_features = load_data()
    
    response = input(f"\n Pokračovat a vytvořit ~{len(all_features)} boxplotů? (ano/ne): ")
    if response.lower() not in ['ano', 'yes', 'a', 'y']:
        return
    
    
    all_results = {}
    
    for i, feature_col in enumerate(acoustic_features, 1):
        mw_results = mann_whitney_tests(df_all, feature_col)
        
        all_results[feature_col] = mw_results
        
        plot_boxplot_with_mw(df_all, feature_col, mw_results, OUTPUT_DIR, is_acoustic=True)
        
        if i % 10 == 0:
            print(f"   {i}/{len(acoustic_features)} akustických featur hotovo")
    
    print(f"   VŠECH {len(acoustic_features)} akustických featur hotovo\n")
    
    for i, feature_col in enumerate(linguistic_features, 1):
        mw_results = mann_whitney_tests(df_all, feature_col)
        
        all_results[feature_col] = mw_results
        
        plot_boxplot_with_mw(df_all, feature_col, mw_results, OUTPUT_DIR, is_acoustic=False)
        
        if i % 5 == 0:
            print(f"   {i}/{len(linguistic_features)} lingvistických featur hotovo")
    
    print(f"   VŠECH {len(linguistic_features)} lingvistických featur hotovo\n")
    
    
    summary = []
    for feature_col, results in all_results.items():
        is_acoustic = feature_col in acoustic_features
        modality = "Akustika" if is_acoustic else "Lingvistika"
        
        for _, row in results.iterrows():
            summary.append({
                'Feature': feature_col,
                'Modality': modality,
                'Group1': row['Group1'],
                'Group2': row['Group2'],
                'median1': row['median1'],
                'median2': row['median2'],
                'p_value': row['p_value'],
                'p_fdr': row['p_fdr'],
                'significant': row['significant'],
                'stars': get_significance_stars(row['p_fdr'])
            })
    
    df_summary = pd.DataFrame(summary)
    summary_file = os.path.join(OUTPUT_DIR, 'mann_whitney_all_features_summary.csv')
    df_summary.to_csv(summary_file, index=False)
    
    
    sig_count = df_summary['significant'].sum()
    print(f"  Celkem testů: {len(df_summary)}")
    print(f"  Signifikantních: {sig_count} ({sig_count/len(df_summary)*100:.1f}%)")
    
    print("HOTOVO!")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
