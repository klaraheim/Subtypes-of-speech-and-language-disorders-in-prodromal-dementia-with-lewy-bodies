# MANN-WHITNEY U test + boxploty (lingvistika)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_FILE = os.path.join(SCRIPT_DIR, "output_clustering_linguistic", "mci_lb_clustery.csv")
COMPOSITE_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_linguistic", "composite_domain_scores.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_mann_whitney_linguistic")

SUBJECT_COL = "Subject"
GROUP_COL = "Category"
HC_LABEL = "HC"
TARGET_LABEL = "MCI-LB"

DOMAIN_NAMES = {
    "Lexikální_score":  "Lexikální",
    "Sémantická_score": "Sémantická",
    "Syntaktická_score": "Syntaktická"
}

ALPHA = 0.05

GROUP_COLORS = {
    0:    '#E8A12A',
    1:    '#C4376B',
    2:    '#7B3FA0',
    3:    '#2B62B8',
    'HC': '#3D5A80',
}

MEDIAN_COLOR  = '#1D1D2E'
MEAN_COLOR    = '#E8582A'
WHISKER_COLOR = '#1D1D2E'

SIG_LINE_COLOR = '#1D1D2E'

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():

    if not os.path.exists(CLUSTER_FILE):
        raise FileNotFoundError(f"Soubor nenalezen: {CLUSTER_FILE}")

    df_clusters = pd.read_csv(CLUSTER_FILE)

    if not os.path.exists(COMPOSITE_FILE):
        raise FileNotFoundError(f"Soubor nenalezen: {COMPOSITE_FILE}")

    df_composite = pd.read_csv(COMPOSITE_FILE)

    df_mcilb_composite = df_composite[df_composite[GROUP_COL] == TARGET_LABEL].copy()
    df_mcilb_with_clusters = pd.merge(
        df_mcilb_composite,
        df_clusters[[SUBJECT_COL, 'cluster']],
        on=SUBJECT_COL,
        how='inner'
    )

    df_hc = df_composite[df_composite[GROUP_COL] == HC_LABEL].copy()
    df_hc['cluster'] = 'HC'

    df_all = pd.concat([df_mcilb_with_clusters, df_hc], ignore_index=True)
    score_cols = [c for c in df_all.columns if c.endswith('_score')]

    return df_all, score_cols

def mann_whitney_tests(df, domain_col):
    unique_groups = df['cluster'].unique()
    numeric_groups = sorted([g for g in unique_groups if g != 'HC'])
    groups = numeric_groups + ['HC']

    results = []

    for g1, g2 in combinations(groups, 2):
        data1 = df[df['cluster'] == g1][domain_col].dropna()
        data2 = df[df['cluster'] == g2][domain_col].dropna()

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

    if len(df_results) > 0:
        _, p_fdr, _, _ = multipletests(df_results['p_value'], alpha=ALPHA, method='fdr_bh')
        df_results['p_fdr'] = p_fdr
    else:
        df_results['p_fdr'] = []
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

def plot_boxplot_with_mw(df, domain_col, domain_name, mw_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    groups = sorted([g for g in df['cluster'].unique() if g != 'HC']) + ['HC']
    data_groups = []
    labels = []
    colors = []

    for g in groups:
        data = df[df['cluster'] == g][domain_col].dropna()
        data_groups.append(data)
        color = GROUP_COLORS.get(g, '#4A90D4')
        colors.append(color)
        if g == 'HC':
            n = len(data)
            labels.append(f'HC\n(n={n})')
        else:
            n = len(data)
            labels.append(f'Shluk {g}\n(n={n})')

    bp = ax.boxplot(
        data_groups,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5),
        meanprops=dict(color=MEAN_COLOR, linewidth=2.5, linestyle='--'),
        whiskerprops=dict(color=WHISKER_COLOR, linewidth=1.5),
        capprops=dict(color=WHISKER_COLOR, linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor=WHISKER_COLOR,
                        markeredgecolor=WHISKER_COLOR, markersize=5, alpha=0.5),
        boxprops=dict(linewidth=1.5)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    for i, (whisker, cap) in enumerate(zip(
        zip(bp['whiskers'][::2], bp['whiskers'][1::2]),
        zip(bp['caps'][::2],     bp['caps'][1::2])
    )):
        c = colors[i]
        for w in whisker:
            w.set_color(c)
        for ca in cap:
            ca.set_color(c)

    for i, flier in enumerate(bp['fliers']):
        flier.set_markerfacecolor(colors[i])
        flier.set_markeredgecolor(colors[i])

    y_vals = [d.max() for d in data_groups if len(d) > 0]
    y_max = max(y_vals)
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

    ax.set_ylabel('Doménové skóre', fontsize=18, fontweight='bold', color='#1D1D2E')
    ax.set_title(f'{domain_name}\n(vyšší hodnota = horší výkon)',
                 fontsize=18, fontweight='bold', color='#1D1D2E')
    ax.grid(axis='y', alpha=0.25, linestyle='--', color='#1D1D2E')
    ax.tick_params(colors='#1D1D2E', labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1D1D2E')
        spine.set_linewidth(0.8)

    legend_elements = [
        Line2D([0], [0], color=MEDIAN_COLOR, linewidth=2.5, label='Medián'),
        Line2D([0], [0], color=MEAN_COLOR,   linewidth=2.5, linestyle='--', label='Průměr'),
    ]
    for g, color in zip(groups, colors):
        label = 'HC' if g == 'HC' else f'Shluk {g}'
        legend_elements.append(
            mpatches.Patch(facecolor=color, alpha=0.55,
                           edgecolor=color, label=label)
        )

    ax.legend(handles=legend_elements, loc='upper right',
              framealpha=0.9, fontsize=16)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f'boxplot_{domain_col}.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)

    df_all, score_cols = load_data()

    all_results = {}

    for domain_col in score_cols:
        domain_name = DOMAIN_NAMES.get(domain_col, domain_col)
        mw_results = mann_whitney_tests(df_all, domain_col)
        all_results[domain_col] = mw_results
        plot_boxplot_with_mw(df_all, domain_col, domain_name, mw_results, OUTPUT_DIR)

    for domain_col, results in all_results.items():
        domain_name = DOMAIN_NAMES.get(domain_col, domain_col)
        output_file = os.path.join(OUTPUT_DIR, f'mw_results_{domain_col}.csv')
        results.to_csv(output_file, index=False)

    summary = []
    for domain_col, results in all_results.items():
        domain_name = DOMAIN_NAMES.get(domain_col, domain_col)
        for _, row in results.iterrows():
            summary.append({
                'Domain': domain_name,
                'Group1': row['Group1'],
                'Group2': row['Group2'],
                'p_value': row['p_value'],
                'p_fdr': row['p_fdr'],
                'significant': row['significant'],
                'stars': get_significance_stars(row['p_fdr'])
            })

    df_summary = pd.DataFrame(summary)
    summary_file = os.path.join(OUTPUT_DIR, 'mann_whitney_summary.csv')
    df_summary.to_csv(summary_file, index=False)

    print(" HOTOVO!")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")
if __name__ == "__main__":
    main()
