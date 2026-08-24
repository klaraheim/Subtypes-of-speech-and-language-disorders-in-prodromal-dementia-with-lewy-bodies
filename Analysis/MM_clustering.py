# Multimodální clustering - kombinace akustika+lingvistika

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLUSTER_COLORS = ['#E8582A', '#C4376B', '#7B3FA0', '#2B62B8', '#E8A12A', '#3D5A80']
COLOR_HC       = '#1D1D2E'
COLOR_INK      = '#1D1D2E'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ACOUSTIC_CLUSTER_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_acoustic", "cluster_input_only_target.csv")
ACOUSTIC_COMPOSITE_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_acoustic", "composite_domain_scores.csv")

LINGUISTIC_CLUSTER_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_linguistic", "cluster_input_only_target.csv")
LINGUISTIC_COMPOSITE_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_linguistic", "composite_domain_scores.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_clustering_multimodal")

SUBJECT_COL = "Subject"
GROUP_COL = "Category"

HC_LABEL = "HC"
TARGET_LABEL = "MCI-LB"

DOMAIN_NAMES = {
    "Respirace_score": "Respirace",
    "Fonace_score": "Fonace",
    "Artikulace_score": "Artikulace",
    "Prozodie_score": "Prozodie",
    "Lexikální_score": "Lexikální",
    "Sémantická_score": "Sémantická",
    "Syntaktická_score": "Syntaktická",
}

ACOUSTIC_OUTLIERS = [15, 35]
LINGUISTIC_OUTLIERS = [40]

MULTIMODAL_OUTLIERS = [
    'Kardiovize_2299',
    'Kardiovize_2488',
    'Kardiovize_854',
    'preDLB_pre-LBD-120#1'
]

N_CLUSTERS = 3
K_RANGE = range(2, 7)
RANDOM_STATE = 42

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_and_merge():

    if os.path.exists(ACOUSTIC_CLUSTER_FILE):
        df_a = pd.read_csv(ACOUSTIC_CLUSTER_FILE)
    else:
        raise FileNotFoundError(f"Soubor nenalezen: {ACOUSTIC_CLUSTER_FILE}")

    if ACOUSTIC_OUTLIERS:
        df_a = df_a.drop(df_a.iloc[ACOUSTIC_OUTLIERS].index).reset_index(drop=True)

    score_cols_a = [c for c in df_a.columns if c.endswith('_score')]

    if os.path.exists(LINGUISTIC_CLUSTER_FILE):
        df_l = pd.read_csv(LINGUISTIC_CLUSTER_FILE)
    else:
        raise FileNotFoundError(f"Soubor nenalezen: {LINGUISTIC_CLUSTER_FILE}")

    if LINGUISTIC_OUTLIERS:
        df_l = df_l.drop(df_l.iloc[LINGUISTIC_OUTLIERS].index).reset_index(drop=True)

    score_cols_l = [c for c in df_l.columns if c.endswith('_score')]

    df_a_scores = df_a[[SUBJECT_COL] + score_cols_a]
    df_l_scores = df_l[[SUBJECT_COL] + score_cols_l]

    df_merged = pd.merge(df_a_scores, df_l_scores, on=SUBJECT_COL, how='inner')
    score_cols = score_cols_a + score_cols_l

    hc_median_profile = None
    try:
        df_all_a = pd.read_csv(ACOUSTIC_COMPOSITE_FILE)
        df_all_l = pd.read_csv(LINGUISTIC_COMPOSITE_FILE)

        hc_a = df_all_a[df_all_a[GROUP_COL] == HC_LABEL][score_cols_a].median()
        hc_l = df_all_l[df_all_l[GROUP_COL] == HC_LABEL][score_cols_l].median()
        hc_median_profile = pd.concat([hc_a, hc_l])
    except Exception as e:
        pass

    return df_merged, score_cols, hc_median_profile

def plot_silhouette_scores(X, k_range, selected_k):
    silhouettes = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), silhouettes, 'o-', linewidth=2, markersize=8, color='#C4376B')

    idx = list(k_range).index(selected_k)
    ax.axvline(selected_k, color='#2B62B8', linestyle='--', linewidth=1.5,
               label=f'Zvolené k = {selected_k}')
    ax.scatter([selected_k], [silhouettes[idx]], color='#2B62B8', zorder=5, s=120)

    ax.set_xlabel('Počet shluků (k)', fontsize=16, color=COLOR_INK)
    ax.set_ylabel('Silhouette Score', fontsize=16, color=COLOR_INK)
    ax.set_title('Silhouette Score pro různá k\n(vyšší = lepší separace shluků)',
                 fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.axhline(0, color=COLOR_INK, linestyle='--', alpha=0.4)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.25, linestyle='--', color=COLOR_INK)
    ax.tick_params(colors=COLOR_INK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_INK)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_scores.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def plot_dendrogram(X):

    Z = linkage(X, method='ward')

    sorted_distances = np.sort(Z[:, 2])[::-1]
    if N_CLUSTERS - 1 < len(sorted_distances):
        threshold = (sorted_distances[N_CLUSTERS - 2] + sorted_distances[N_CLUSTERS - 1]) / 2
    else:
        threshold = 0.7 * max(Z[:, 2])

    plt.figure(figsize=(14, 8))

    dendrogram(
        Z,
        leaf_font_size=8,
        color_threshold=threshold,
        no_labels=True
    )

    plt.axhline(y=threshold, color=COLOR_INK, linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'Řez pro {N_CLUSTERS} shluky')
    plt.legend(fontsize=16, loc='upper right')

    plt.title('Multimodální shluky MCI-LB subjektů\n(akustické + lingvistické domény)',
              fontsize=18, fontweight='bold', color=COLOR_INK)
    plt.grid(axis='y', alpha=0.25, linestyle='--', color=COLOR_INK)
    plt.gca().tick_params(colors=COLOR_INK, labelsize=16)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(COLOR_INK)
        spine.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'MM_dendrogram.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def radar_plot_with_hc_norm(cluster_medians, hc_median_profile, score_cols):
    domain_names = [DOMAIN_NAMES.get(col, col.replace('_score', ''))
                    for col in score_cols]
    n_axes = len(domain_names)

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(13, 13))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], domain_names, fontsize=16)

    all_values = cluster_medians.values.flatten().tolist()
    if hc_median_profile is not None:
        all_values += hc_median_profile.values.tolist()

    vmin = float(np.floor(min(all_values) - 0.5))
    vmax = float(np.ceil(max(all_values) + 0.5))
    if vmin == vmax:
        vmin, vmax = -1.0, 2.0

    yticks = np.linspace(vmin, vmax, 5)
    ax.set_rlabel_position(0)
    plt.yticks(yticks, [f"{y:.1f}" for y in yticks], fontsize=16, color=COLOR_INK)
    plt.ylim(vmin, vmax)

    for cluster_id in cluster_medians.index:
        vals = cluster_medians.loc[cluster_id].tolist()
        vals += vals[:1]

        color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]

        ax.fill(angles, vals, alpha=0.12, color=color, zorder=3)
        ax.fill(angles, vals, alpha=0.12, color=color, zorder=2)
        ax.fill(angles, vals, alpha=0.12, color=color, zorder=1)

        ax.plot(angles, vals,
                linestyle='-',
                linewidth=3,
                color=color,
                label=f"Shluk {cluster_id}",
                marker='o',
                markersize=6,
                zorder=4)

    if hc_median_profile is not None:
        hc_vals = hc_median_profile.tolist()
        hc_vals += hc_vals[:1]

        ax.plot(angles, hc_vals,
                linestyle='--',
                linewidth=2.5,
                color=COLOR_HC,
                marker='o',
                markersize=6,
                markerfacecolor=COLOR_HC,
                markeredgecolor='white',
                markeredgewidth=1.5,
                label='HC medián',
                zorder=5,
                alpha=0.9)

    ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray', linewidth=1)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray', linewidth=0.5)

    ax.spines['polar'].set_color(COLOR_INK)
    ax.spines['polar'].set_linewidth(1.5)

    plt.title("Multimodální profil domén v shlukech MCI-LB\n(mediány, akustické + lingvistické; vyšší = horší)",
              y=1.08, fontsize=18, fontweight='bold', color=COLOR_INK)

    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
               fontsize=16, frameon=True, framealpha=0.95, edgecolor='#bdc3c7')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "MM_radar.pdf"),
                format='pdf', bbox_inches='tight')
    plt.close()

def plot_pca_clusters(X, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    for cluster in np.unique(labels):
        mask = labels == cluster
        color = CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            color=color,
            label=f"Shluk {cluster}",
            s=120,
            alpha=0.7,
            edgecolors=COLOR_INK,
            linewidth=0.8
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    plt.xlabel(f"PC1 ({var1:.1f}%)", fontsize=16, color=COLOR_INK)
    plt.ylabel(f"PC2 ({var2:.1f}%)", fontsize=16, color=COLOR_INK)
    plt.title("Multimodální shlukování MCI-LB pacientů (akustické + lingvistické domény)",
              fontsize=18, fontweight='bold', color=COLOR_INK)
    plt.legend(title="Shluky", fontsize=16)
    plt.grid(alpha=0.25, linestyle='--', color=COLOR_INK)
    plt.gca().tick_params(colors=COLOR_INK, labelsize=16)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor(COLOR_INK)
        spine.set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "MM_shluky_PCA.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

def plot_cluster_sizes(labels):
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        cluster_counts.index,
        cluster_counts.values,
        color=[CLUSTER_COLORS[int(i) % len(CLUSTER_COLORS)] for i in cluster_counts.index],
        edgecolor=COLOR_INK,
        linewidth=1.2,
        alpha=0.85
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=18, fontweight='bold', color=COLOR_INK)

    ax.set_xlabel('Shluk', fontsize=16, color=COLOR_INK)
    ax.set_ylabel('Počet MCI-LB pacientů', fontsize=16, color=COLOR_INK)
    ax.set_title('Velikost shluků (MCI-LB pacienti)', fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_xticks(cluster_counts.index)
    ax.set_xticklabels([f'Shluk {int(i)}' for i in cluster_counts.index])
    ax.grid(axis='y', alpha=0.25, linestyle='--', color=COLOR_INK)
    ax.tick_params(colors=COLOR_INK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_INK)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'velikost_shluku.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)

    df_merged, score_cols, hc_median_profile = load_and_merge()
    
    if MULTIMODAL_OUTLIERS:
        df_merged = df_merged[~df_merged[SUBJECT_COL].isin(MULTIMODAL_OUTLIERS)].reset_index(drop=True)

    X = df_merged[score_cols].to_numpy(dtype=float)

    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]
    df_clean = df_merged[valid_mask].copy()

    if len(df_clean) < 10:
        return

    best_k = N_CLUSTERS

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_clean)

    df_clean = df_clean.copy()
    df_clean['cluster'] = labels

    sil_final = silhouette_score(X_clean, labels)

    cluster_medians = df_clean.groupby('cluster')[score_cols].median()

    for cluster_id in sorted(df_clean['cluster'].unique()):
        n = (df_clean['cluster'] == cluster_id).sum()
        pct = n / len(df_clean) * 100

        for col in score_cols:
            val = cluster_medians.loc[cluster_id, col]
            domain = DOMAIN_NAMES.get(col, col.replace('_score', ''))

    plot_silhouette_scores(X_clean, K_RANGE, best_k)
    plot_dendrogram(X_clean)
    radar_plot_with_hc_norm(cluster_medians, hc_median_profile, score_cols)
    plot_pca_clusters(X_clean, labels)
    plot_cluster_sizes(labels)

    df_clean.to_csv(os.path.join(OUTPUT_DIR, 'mci_lb_clustery.csv'), index=False)
    cluster_medians.to_csv(os.path.join(OUTPUT_DIR, 'mediany_clusteru.csv'))

    print(" HOTOVO!")
    print(f"\nAnalyzováno: {len(df_clean)} MCI-LB pacientů")
    print(f"Počet shluků: {best_k}")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
