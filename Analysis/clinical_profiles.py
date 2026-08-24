# Klinické profily podle kombinace akustických a lingvistických shluků

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal
from itertools import combinations

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLOR_INK  = '#1D1D2E'
COLOR_ORANGE = '#E8582A'

COMBO_COLORS = [
    '#2B62B8',
    '#C4376B',
    '#7B3FA0',
    '#E8582A',
    '#E8A12A',
    '#3D5A80',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ACOUSTIC_CLUSTERS   = os.path.join(SCRIPT_DIR, "output_clustering_acoustic", "mci_lb_clustery.csv")
LINGUISTIC_CLUSTERS = os.path.join(SCRIPT_DIR, "output_clustering_linguistic", "mci_lb_clustery.csv")
METADATA_FILE       = os.path.join(SCRIPT_DIR, "metadata_altered_plusMedian.xlsx")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_clinical_profiles")

SUBJECT_COL = "Subject"

CLINICAL_VARS = {
    'age':                  'Věk',
    'education length':     'Délka vzdělání',
    'MOCA':                 'MoCA',
    'GDS':                  'GDS',
    'memory_zscore':        'Paměť (z-score)',
    'visuospatial_zscore':  'Visuospatial (z-score)',
    'attention_zscore':     'Pozornost (z-score)',
    'executive_zscore':     'Exekutivní (z-score)',
}

ALPHA = 0.05

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():

    df_a = pd.read_csv(ACOUSTIC_CLUSTERS)
    df_l = pd.read_csv(LINGUISTIC_CLUSTERS)
    df_meta = pd.read_excel(METADATA_FILE, engine='openpyxl')

    df_a = df_a[[SUBJECT_COL, 'cluster']].rename(columns={'cluster': 'cluster_A'})
    df_l = df_l[[SUBJECT_COL, 'cluster']].rename(columns={'cluster': 'cluster_L'})

    df = pd.merge(df_a, df_l, on=SUBJECT_COL, how='inner')
    df = pd.merge(df, df_meta, on=SUBJECT_COL, how='inner')

    df['combo'] = df.apply(lambda r: f"A{r['cluster_A']}+L{r['cluster_L']}", axis=1)

    return df

def kruskal_wallis_tests(df, clinical_vars):

    results = []
    combos = sorted(df['combo'].unique())

    for var, label in clinical_vars.items():
        if var not in df.columns:
            continue

        groups = [df[df['combo'] == c][var].dropna().values for c in combos]
        groups = [g for g in groups if len(g) >= 3]

        if len(groups) < 2:
            continue

        stat, p = kruskal(*groups)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

        results.append({
            'Variable': label,
            'H_statistic': stat,
            'p_value': p,
            'significant': p < ALPHA,
            'stars': sig
        })

    return pd.DataFrame(results)

def plot_clinical_profiles(df, clinical_vars):
    combos = sorted(df['combo'].unique())
    colors = {c: COMBO_COLORS[i % len(COMBO_COLORS)] for i, c in enumerate(combos)}

    available = [(var, label) for var, label in clinical_vars.items() if var in df.columns]

    legend_elements = [
        mpatches.Patch(facecolor=colors[c], alpha=0.55, edgecolor=colors[c], label=c)
        for c in combos
    ] + [
        plt.Line2D([0], [0], color=COLOR_INK,    linewidth=2.5, label='Medián'),
        plt.Line2D([0], [0], color=COLOR_ORANGE, linewidth=2.5, linestyle='--', label='Průměr'),
    ]

    chunks = [available[i:i+2] for i in range(0, len(available), 2)]

    for part_idx, chunk in enumerate(chunks):
        n_plots = len(chunk)
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
        if n_plots == 1:
            axes = [axes]

        for idx, (var, label) in enumerate(chunk):
            ax = axes[idx]
            data_groups = [df[df['combo'] == c][var].dropna().values for c in combos]
            clrs = [colors[c] for c in combos]

            bp = ax.boxplot(
                data_groups,
                tick_labels=combos,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                medianprops=dict(color=COLOR_INK,    linewidth=2.5),
                meanprops=dict(color=COLOR_ORANGE,   linewidth=2.5, linestyle='--'),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=5, alpha=0.5),
                boxprops=dict(linewidth=1.5)
            )

            for patch, c in zip(bp['boxes'], clrs):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
                patch.set_edgecolor(c)

            for i, (w0, w1, c0, c1) in enumerate(zip(
                bp['whiskers'][::2], bp['whiskers'][1::2],
                bp['caps'][::2],     bp['caps'][1::2]
            )):
                c = clrs[i]
                w0.set_color(c); w1.set_color(c)
                c0.set_color(c); c1.set_color(c)

            for i, flier in enumerate(bp['fliers']):
                flier.set_markerfacecolor(clrs[i])
                flier.set_markeredgecolor(clrs[i])

            ax.set_title(label, fontsize=18, fontweight='bold', color=COLOR_INK)
            ax.set_ylabel('Hodnota', fontsize=16, color=COLOR_INK)
            ax.set_xlabel('Kombinace shluků (Akustický + Lingvistický)', fontsize=16, color=COLOR_INK)
            ax.grid(axis='y', alpha=0.25, linestyle='--', color=COLOR_INK)
            ax.tick_params(colors=COLOR_INK, labelsize=16)
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_INK)
                spine.set_linewidth(0.8)

        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=len(combos) + 2, fontsize=16, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.12))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        out_path = os.path.join(OUTPUT_DIR, f'clinical_profiles_boxplots_{part_idx+1}.pdf')
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"   Uloženo: {out_path}")

def main():
    ensure_dir(OUTPUT_DIR)

    df = load_data()

    kw_results = kruskal_wallis_tests(df, CLINICAL_VARS)
    kw_results.to_csv(os.path.join(OUTPUT_DIR, 'kruskal_wallis_results.csv'), index=False)

    available_vars = [v for v in CLINICAL_VARS if v in df.columns]
    summary = df.groupby('combo')[available_vars].median().round(2)
    summary.columns = [CLINICAL_VARS[v] for v in summary.columns]
    summary.to_csv(os.path.join(OUTPUT_DIR, 'clinical_profiles_summary.csv'))
    print(f"\n Uloženo: clinical_profiles_summary.csv")

    plot_clinical_profiles(df, CLINICAL_VARS)

    print("HOTOVO!")
    print(f"Počet subjektů v analýze: {len(df)}")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
