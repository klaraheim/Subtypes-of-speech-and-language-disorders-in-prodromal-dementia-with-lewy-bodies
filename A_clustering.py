# Akustický clustering

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


COLORS = ['#E8582A', '#C4376B', '#7B3FA0', '#2B62B8', '#E8A12A', '#1D1D2E']
COLORS_SECONDARY = ['#F2856A', '#E84078', '#5B2D82', '#4A90D4', '#F5C842', '#3D5A80']
COLORS_ALL = COLORS + COLORS_SECONDARY

COLOR_HC_MEDIAN = '#1D1D2E'


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMPOSITE_SCORES_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_acoustic", "composite_domain_scores.csv")
CLUSTER_INPUT_FILE = os.path.join(SCRIPT_DIR, "output_domain_scores_acoustic", "cluster_input_only_target.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_clustering_acoustic")

SUBJECT_COL = "Subject"
GROUP_COL = "Category"

HC_LABEL = "HC"
TARGET_LABEL = "MCI-LB"

DOMAIN_NAMES = {
    "Respirace_score": "Respirace",
    "Fonace_score": "Fonace",
    "Artikulace_score": "Artikulace",
    "Prozodie_score": "Prozodie"
}

N_CLUSTERS = 3
K_RANGE = range(2, 7)
RANDOM_STATE = 42

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():
    
    if os.path.exists(CLUSTER_INPUT_FILE):
        df_mcilb = pd.read_csv(CLUSTER_INPUT_FILE)
    else:
        raise FileNotFoundError(f"Soubor nenalezen: {CLUSTER_INPUT_FILE}")
    
    if os.path.exists(COMPOSITE_SCORES_FILE):
        df_all = pd.read_csv(COMPOSITE_SCORES_FILE)
    else:
        df_all = None
    
    score_cols = [c for c in df_mcilb.columns if c.endswith('_score')]
    
    return df_mcilb, df_all, score_cols

def plot_silhouette_scores(X, k_range, selected_k):
    silhouettes = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), silhouettes, 'o-', linewidth=2, markersize=8, color=COLORS[0])

    idx = list(k_range).index(selected_k)
    ax.axvline(selected_k, color=COLORS[3], linestyle='--', linewidth=1.5,
               label=f'Zvolené k = {selected_k}')
    ax.scatter([selected_k], [silhouettes[idx]], color=COLORS[3], zorder=5, s=120)

    ax.set_xlabel('Počet shluků (k)', fontsize=16)
    ax.set_ylabel('Silhouette Score', fontsize=16)
    ax.set_title('Silhouette Score pro různá k\n(vyšší = lepší separace shluků)',
                 fontsize=18, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'A_silhouette_scores.pdf'), dpi=300)
    plt.close()

def plot_dendrogram(X):
    
    Z = linkage(X, method='ward')
    
    plt.figure(figsize=(14, 8))
    
    dendrogram(
        Z,
        leaf_font_size=8,
        color_threshold=0.7*max(Z[:,2]),
        no_labels=True
    )
    
    plt.title('Shluky MCI-LB subjektů podle akustických parametrů', 
              fontsize=18, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'A_dendrogram.pdf'), dpi=300)
    plt.close()
    

def radar_plot_with_hc_norm(cluster_medians, hc_median_profile, score_cols):
    domain_names = [DOMAIN_NAMES.get(col, col.replace('_score', '')) 
                    for col in score_cols]
    n_axes = len(domain_names)

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(12, 12))
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
    plt.yticks(yticks, [f"{y:.1f}" for y in yticks], fontsize=16)
    plt.ylim(vmin, vmax)

    colors = COLORS

    for cluster_id in cluster_medians.index:
        vals = cluster_medians.loc[cluster_id].tolist()
        vals += vals[:1]
        
        color = colors[int(cluster_id) % len(colors)]

        ax.fill(angles, vals, alpha=0.15, color=color)
        ax.plot(angles, vals, 'o-', linewidth=2.5, 
                label=f"Shluk {cluster_id}", 
                color=color, markersize=8)

    if hc_median_profile is not None:
        hc_vals = hc_median_profile.tolist()
        hc_vals += hc_vals[:1]
        
        ax.plot(angles, hc_vals, 
                linestyle='--',
                linewidth=3,
                color=COLOR_HC_MEDIAN,
                marker='o',
                markersize=10,
                markerfacecolor=COLOR_HC_MEDIAN,
                markeredgecolor='white',
                markeredgewidth=2,
                label='HC medián',
                zorder=10)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.xaxis.grid(True, linestyle='-', alpha=0.4, color='gray')
    
    plt.title("Profil akustických domén v shlucích MCI-LB\n(mediány, vyšší hodnota = horší výkon)", 
              y=1.08, fontsize=18, fontweight='bold')
    
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), 
               fontsize=16, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "A_radar.pdf"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_pca_clusters(X, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    colors = COLORS
    
    for cluster in np.unique(labels):
        mask = labels == cluster
        color = colors[int(cluster) % len(colors)]
        
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            color=color,
            label=f"Shluk {cluster}",
            s=120,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.8
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    plt.xlabel(f"PC1 ({var1:.1f}%)", fontsize=16)
    plt.ylabel(f"PC2 ({var2:.1f}%)", fontsize=16)
    plt.title("Shluky MCI-LB subjektů podle akustických parametrů", 
              fontsize=18, fontweight='bold')
    plt.legend(title="Shluky", fontsize=16)
    plt.grid(alpha=0.3)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "A_shluky_PCA.pdf"), dpi=300)
    plt.close()
    

def plot_cluster_sizes(labels):
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = COLORS
    
    bars = ax.bar(
        cluster_counts.index, 
        cluster_counts.values,
        color=[colors[int(i) % len(colors)] for i in cluster_counts.index],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    ax.set_xlabel('Shluk', fontsize=16)
    ax.set_ylabel('Počet MCI-LB pacientů', fontsize=16)
    ax.set_title('Velikost shluků (MCI-LB pacienti)', fontsize=18, fontweight='bold')
    ax.set_xticks(cluster_counts.index)
    ax.set_xticklabels([f'Shluk {int(i)}' for i in cluster_counts.index])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'A_velikost_clusteru.pdf'), dpi=300)
    plt.close()
    

def main():
    ensure_dir(OUTPUT_DIR)

    df_mcilb, df_all, score_cols = load_data()

    to_remove = [15, 34]
    df_mcilb = df_mcilb.drop(df_mcilb.iloc[to_remove].index).reset_index(drop=True)

    X = df_mcilb[score_cols].to_numpy(dtype=float)
    
    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]
    df_clean = df_mcilb[valid_mask].copy()
    

    if len(df_clean) < 10:
        return

    best_k = N_CLUSTERS

    
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_clean)
    
    df_clean['cluster'] = labels
    
    sil_final = silhouette_score(X_clean, labels)

    
    
    cluster_medians = df_clean.groupby('cluster')[score_cols].median()
    
    hc_median_profile = None
    if df_all is not None and GROUP_COL in df_all.columns:
        hc_median_profile = df_all[df_all[GROUP_COL] == HC_LABEL][score_cols].median()

    
    plot_silhouette_scores(X_clean, K_RANGE, best_k)
    plot_dendrogram(X_clean)
    radar_plot_with_hc_norm(cluster_medians, hc_median_profile, score_cols)
    plot_pca_clusters(X_clean, labels)
    plot_cluster_sizes(labels)

    df_clean.to_csv(os.path.join(OUTPUT_DIR, 'mci_lb_shluky.csv'), index=False)
    
    cluster_medians.to_csv(os.path.join(OUTPUT_DIR, 'mediany_shluku.csv'))

    print("HOTOVO!")
    print(f"\nAnalyzováno: {len(df_clean)} MCI-LB pacientů")
    print(f"Počet shluků: {best_k}")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
