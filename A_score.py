# Akustická doménová skóre

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


COLOR_HC     = '#3D5A80'
COLOR_TARGET = '#C4376B'
COLOR_INK    = '#1D1D2E'
COLOR_ORANGE   = '#E8582A'

BAR_COLORS = ['#2B62B8', '#4A90D4', '#7B3FA0', '#C4376B', '#E8582A',
              '#E8A12A', '#F5C842', '#3D5A80', '#5B2D82', '#F2856A',
              '#E84078', '#1D1D2E']


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "acoustic_residualized_filtered.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_domain_scores_acoustic")

SUBJECT_COL = "Subject"
GROUP_COL = "Category"
HC_LABEL = "HC"
TARGET_LABEL = "MCI-LB"

DOMAINS = {
    "Respirace": [
        "RR_TSK2", "IPAmean_TSK2", "MPT_TSK6"
    ],
    
    "Fonace": [
        "HNR_TSK3", "HNR_TSK4", "HNR_TSK5", "HNR_TSK6",
        "HRF_TSK3", "HRF_TSK4", "HRF_TSK5", "HRF_TSK6",
        "Jitter_TSK3", "Jitter_TSK4", "Jitter_TSK5", "Jitter_TSK6",
        "Shimmer_TSK3", "Shimmer_TSK4", "Shimmer_TSK5", "Shimmer_TSK6",
        "PPQ_TSK3", "PPQ_TSK4", "PPQ_TSK5", "PPQ_TSK6",
        "APQ_TSK3", "APQ_TSK4", "APQ_TSK5", "APQ_TSK6",
        "DF0_TSK3", "DF0_TSK4", "DF0_TSK5", "DF0_TSK6",
        "DDF0_TSK3", "DDF0_TSK4", "DDF0_TSK5", "DDF0_TSK6",
        "DUV_TSK3", "DUV_TSK4", "DUV_TSK5", "DUV_TSK6",
        "RelF1SD_TSK3", "RelF1SD_TSK4", "RelF1SD_TSK5", "RelF1SD_TSK6",
        "RelF2SD_TSK3", "RelF2SD_TSK4", "RelF2SD_TSK5", "RelF2SD_TSK6"
    ],
    
    "Artikulace": [
        "RelF1SD_TSK1", "RelF1SD_TSK2", 
        "RelF2SD_TSK1", "RelF2SD_TSK2", 
        "VAI_TSK1", "VAI_TSK2",
        "VSHA_TSK1", "VSHA_TSK2",
        "AAVS_TSK1", "AAVS_TSK2",
        "PR_TSK7", "COV_TSK7", "PA_TSK7", "RI_TSK7", "RA_TSK7"
    ],
    
    "Prozodie": [
        "RelF0SD_TSK1", "RelF0SD_TSK2",
        "RelSE0SD_TSK1", "RelSE0SD_TSK2",
        "DurMED_TSK1", "DurMED_TSK2",
        "DurMAD_TSK1", "DurMAD_TSK2",
        "SPIR_TSK1", "SPIR_TSK2",
        "pause_n_TSK1", "pause_n_TSK2",
        "pause_ratio_TSK1", "pause_ratio_TSK2"
    ]
}

RANDOM_STATE = 42

print(os.path.abspath(INPUT_FILE))
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def hc_percentile_norm(df, features, group_col, hc_label):
 
    hc_mask = df[group_col] == hc_label
    
    if hc_mask.sum() == 0:
        raise ValueError(f"Žádné subjekty se skupinou '{hc_label}'")
    
    medians = df.loc[hc_mask, features].median()
    p95s = df.loc[hc_mask, features].quantile(0.95)
    
    diffs = p95s - medians
    
    valid = diffs.replace(0, np.nan).dropna().index.tolist()
    medians = medians[valid]
    p95s = p95s[valid]
    diffs = diffs[valid]
    
    X_norm = (df[valid] - medians) / diffs
    
    return X_norm, medians, p95s, diffs, valid

def calculate_feature_auc_and_flip(df, feature, X_norm, group_col, hc_label, target_label):

    # Pokud je koeficient < 0, otočí featuru
    
    df_temp = df[[feature, group_col]].copy()
    df_temp['f_norm'] = X_norm[feature]
    df_temp = df_temp.dropna()
    df_temp = df_temp[df_temp[group_col].isin([hc_label, target_label])]
    
    if len(df_temp) < 10:
        return None, 0.0, False
    
    X_feature = df_temp['f_norm'].values.reshape(-1, 1)
    y = (df_temp[group_col] == target_label).astype(int).values
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_feature, y)
    
    coef = model.coef_[0][0]
    
    flipped = False
    if coef < 0:
        X_norm[feature] = -X_norm[feature]
        flipped = True
        
        df_temp['f_norm'] = X_norm.loc[df_temp.index, feature]
        X_feature = df_temp['f_norm'].values.reshape(-1, 1)
        model.fit(X_feature, y)
        coef = model.coef_[0][0]
    
    y_pred = model.predict_proba(X_feature)[:, 1]
    auc = roc_auc_score(y, y_pred)
    
    weight = auc if auc >= 0.5 else 0.0
    
    return auc, weight, flipped

def calculate_domain_score(df, domain_name, features, group_col, hc_label, target_label):

    available = [f for f in features if f in df.columns]
    
    if len(available) == 0:
        return None
    
    
    X_norm, medians, p95s, diffs, valid_features = hc_percentile_norm(
        df, available, group_col, hc_label
    )
    
    
    
    aucs = []
    weights = []
    features_flipped = []
    
    for feature in valid_features:
        auc, weight, flipped = calculate_feature_auc_and_flip(
            df, feature, X_norm, group_col, hc_label, target_label
        )
        
        if auc is None:
            continue
        
        aucs.append(auc)
        weights.append(weight)
        
        flip_mark = " → OTOČENO!" if flipped else ""
        
        if flipped:
            features_flipped.append(feature)
    
    if len(weights) == 0:
        return None
    
    sum_weights = sum(weights)
    
    if sum_weights == 0:
        return None
    
    domain_scores = []
    for idx in df.index:
        weighted_sum = 0
        for i, feature in enumerate(valid_features[:len(weights)]):
            if idx in X_norm.index:
                f_val = X_norm.loc[idx, feature]
                if not pd.isna(f_val):
                    weighted_sum += weights[i] * f_val
        
        score = weighted_sum / sum_weights
        domain_scores.append(score)
    
    
    return {
        'scores': domain_scores,
        'weights': weights,
        'aucs': aucs,
        'features': valid_features[:len(weights)],
        'features_flipped': features_flipped,
        'medians': medians,
        'p95s': p95s,
        'diffs': diffs
    }

def visualize_weights(weights_df, domain_name, output_dir):
    if len(weights_df) == 0:
        return None

    top = weights_df.head(15).copy()
    n   = len(top)

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('grad', ['#C4376B', '#4A90D4'])
    colors = [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))

    ax.barh(range(n), top['weight'], color=colors, edgecolor='none', alpha=0.85)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top['feature'], fontsize=16, color=COLOR_INK)
    ax.set_xlabel('Váha', fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.set_title(f'{domain_name} – Top parametry podle váhy',
                 fontsize=18, fontweight='bold', color=COLOR_INK)
    ax.grid(axis='x', alpha=0.25, linestyle='--', color=COLOR_INK)
    ax.tick_params(colors=COLOR_INK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_INK)
        spine.set_linewidth(0.8)

    plt.tight_layout()

    filename = f"{domain_name}_weights.pdf"
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()

    return filename

def visualize_domain_scores(out_scores, group_col, hc_label, target_label, output_dir):
    score_cols = [c for c in out_scores.columns if c.endswith('_score')]

    if len(score_cols) == 0:
        return

    n_domains = len(score_cols)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    if n_domains == 1:
        axes = [axes]

    group_color = {hc_label: COLOR_HC, target_label: COLOR_TARGET}

    for idx, score_col in enumerate(score_cols):
        ax          = axes[idx]
        domain_name = score_col.replace('_score', '')
        groups      = [target_label, hc_label]

        data_to_plot = [
            out_scores[out_scores[group_col] == g][score_col].dropna().values
            for g in groups
        ]
        colors = [group_color.get(g, '#4A90D4') for g in groups]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=groups,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops=dict(color=COLOR_INK,  linewidth=2.5),
            meanprops=dict(color=COLOR_ORANGE,   linewidth=2.5, linestyle='--'),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker='o', markersize=5, alpha=0.5),
            boxprops=dict(linewidth=1.5)
        )

        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(c)

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

        d0, d1 = data_to_plot[0], data_to_plot[1]
        if len(d0) >= 3 and len(d1) >= 3:
            _, pval = mannwhitneyu(d0, d1, alternative='two-sided')
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else:
                stars = 'ns'

            all_vals = np.concatenate([d0, d1])
            y_min_data = np.nanmin(all_vals)
            y_max_data = np.nanmax(all_vals)
            y_span   = y_max_data - y_min_data if y_max_data != y_min_data else 1.0
            y_line   = y_max_data + y_span * 0.08
            y_text   = y_line     + y_span * 0.04
            ax.plot([1, 1, 2, 2], [y_line - y_span*0.02, y_line,
                                    y_line, y_line - y_span*0.02],
                    color=COLOR_INK, linewidth=1.2)
            color_star = COLOR_INK if stars == 'ns' else COLOR_TARGET
            ax.text(1.5, y_text, stars, ha='center', va='bottom',
                    fontsize=16 if stars != 'ns' else 13,
                    fontweight='bold', color=color_star)
            ax.set_ylim(top=y_text + y_span * 0.08)

        ax.axhline(0, color=COLOR_INK, linestyle='--', linewidth=1, alpha=0.4)
        ax.set_ylabel('Kompozitní skóre\n(vyšší = horší)', fontsize=16, color=COLOR_INK)
        ax.set_title(domain_name, fontsize=18, fontweight='bold', color=COLOR_INK)
        ax.grid(axis='y', alpha=0.25, linestyle='--', color=COLOR_INK)
        ax.tick_params(colors=COLOR_INK, labelsize=16)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_INK)
            spine.set_linewidth(0.8)

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HC,     alpha=0.55, edgecolor=COLOR_HC,     label='HC'),
        mpatches.Patch(facecolor=COLOR_TARGET,  alpha=0.55, edgecolor=COLOR_TARGET, label='MCI-LB'),
        Line2D([0], [0], color=COLOR_INK,  linewidth=2.5, label='Medián'),
        Line2D([0], [0], color=COLOR_ORANGE, linewidth=2.5, linestyle='--', label='Průměr'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=16, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'domain_scores_boxplots.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Soubor nenalezen: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    
    
    out_scores = df[[SUBJECT_COL, GROUP_COL]].copy()
    
    weights_writer = pd.ExcelWriter(
        os.path.join(OUTPUT_DIR, 'domain_weights.xlsx'), 
        engine='openpyxl'
    )
    norms_writer = pd.ExcelWriter(
        os.path.join(OUTPUT_DIR, 'hc_norm_params.xlsx'), 
        engine='openpyxl'
    )
    
    summary_rows = []
    
    for domain_name, features in DOMAINS.items():
        result = calculate_domain_score(
            df, domain_name, features, GROUP_COL, HC_LABEL, TARGET_LABEL
        )
        
        if result is None:
            continue
        
        out_scores[f'{domain_name}_score'] = result['scores']
        
        weights_df = pd.DataFrame({
            'feature': result['features'],
            'AUC': result['aucs'],
            'weight': result['weights'],
            'flipped': [f in result['features_flipped'] for f in result['features']]
        }).sort_values('weight', ascending=False)
        
        weights_df.to_excel(weights_writer, sheet_name=domain_name[:31], index=False)
        
        norms_df = pd.DataFrame({
            'feature': result['features'],
            'hc_median': result['medians'].loc[result['features']].values,
            'hc_p95': result['p95s'].loc[result['features']].values,
            'hc_diff': result['diffs'].loc[result['features']].values
        })
        
        norms_df.to_excel(norms_writer, sheet_name=domain_name[:31], index=False)
        
        visualize_weights(weights_df, domain_name, OUTPUT_DIR)

        summary_rows.append({
            'domain': domain_name,
            'n_features': len(result['features']),
            'n_flipped': len(result['features_flipped']),
            'mean_weight': np.mean(result['weights']),
            'mean_auc': np.mean(result['aucs'])
        })
    
    weights_writer.close()
    norms_writer.close()
    
    
    visualize_domain_scores(out_scores, GROUP_COL, HC_LABEL, TARGET_LABEL, OUTPUT_DIR)
    
    out_scores.to_csv(
        os.path.join(OUTPUT_DIR, 'composite_domain_scores.csv'),
        index=False
    )
    
    score_cols = [c for c in out_scores.columns if c.endswith('_score')]
    cluster_input = out_scores[out_scores[GROUP_COL] == TARGET_LABEL][
        [SUBJECT_COL, GROUP_COL] + score_cols
    ].copy()
    
    cluster_input.to_csv(
        os.path.join(OUTPUT_DIR, 'cluster_input_only_target.csv'),
        index=False
    )
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, 'domain_summary.csv'),
        index=False
    )
    
    print("HOTOVO!")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
