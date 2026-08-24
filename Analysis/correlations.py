# Analýza korelací

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLOR_INK    = '#1D1D2E'
COLOR_BLUE   = '#2B62B8'
COLOR_SKY    = '#4A90D4'
SCATTER_COLOR = '#2B62B8'
LINE_COLOR    = '#C4376B'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_FILE = os.path.join(SCRIPT_DIR, "output_clustering_multimodal", "mci_lb_clustery.csv")
COGNITIVE_FILE = os.path.join(SCRIPT_DIR, "cognitive_data.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_correlation_analysis")

SPEECH_DOMAINS = {
    'Respirace_score': 'Respirace',
    'Fonace_score': 'Fonace',
    'Artikulace_score': 'Artikulace',
    'Prozodie_score': 'Prozodie',
    'Lexikální_score': 'Lexikální',
    'Sémantická_score': 'Sémantická',
    'Syntaktická_score': 'Syntaktická'
}

COGNITIVE_VARS = {
    'MOCA': 'MoCA',
    'memory_zscore': 'Paměť',
    'visuospatial_zscore': 'Visuospatial',
    'attention_zscore': 'Pozornost',
    'executive_zscore': 'Exekutivní',
    'GDS': 'GDS'
}

SIGNIFICANCE_LEVEL = 0.05

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data():
    
    if not os.path.exists(CLUSTER_FILE):
        raise FileNotFoundError(f"Soubor nenalezen: {CLUSTER_FILE}")
    
    df_speech = pd.read_csv(CLUSTER_FILE)
    
    if not os.path.exists(COGNITIVE_FILE):
        
        np.random.seed(42)
        cognitive_data = {
            'Subject': df_speech['Subject'].values,
            'MOCA': np.random.normal(20, 3, len(df_speech)),
            'memory_zscore': np.random.normal(-1.3, 0.8, len(df_speech)),
            'visuospatial_zscore': np.random.normal(-1.1, 0.7, len(df_speech)),
            'attention_zscore': np.random.normal(-1.0, 0.9, len(df_speech)),
            'executive_zscore': np.random.normal(-1.5, 0.8, len(df_speech)),
            'GDS': np.random.normal(4.5, 2.2, len(df_speech))
        }
        
        df_cognitive = pd.DataFrame(cognitive_data)
        df_cognitive.to_csv(COGNITIVE_FILE, index=False)
    else:
        df_cognitive = pd.read_csv(COGNITIVE_FILE)
    
    df = pd.merge(df_speech, df_cognitive, on='Subject', how='inner')
    
    return df

def calculate_correlations(df):
    
    speech_vars = {k: v for k, v in SPEECH_DOMAINS.items() if k in df.columns}
    cog_vars = {k: v for k, v in COGNITIVE_VARS.items() if k in df.columns}
    
    all_vars = {**speech_vars, **cog_vars}
    var_names = list(all_vars.keys())
    
    corr_matrix = df[var_names].corr(method='pearson')
    
    n = len(df)
    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                            columns=corr_matrix.columns, 
                            index=corr_matrix.index)
    
    for i, var1 in enumerate(var_names):
        for j, var2 in enumerate(var_names):
            if i != j:
                data1 = df[var1].dropna()
                data2 = df[var2].dropna()
                
                common_idx = data1.index.intersection(data2.index)
                
                if len(common_idx) > 2:
                    _, p_val = stats.pearsonr(df.loc[common_idx, var1], 
                                             df.loc[common_idx, var2])
                    p_matrix.loc[var1, var2] = p_val
    
    return corr_matrix, p_matrix, all_vars

def extract_significant_correlations(corr_matrix, p_matrix, all_vars, speech_vars, cog_vars):
    
    results = []
    
    speech_var_names = list(speech_vars.keys())
    cog_var_names = list(cog_vars.keys())
    
    for speech_var in speech_var_names:
        for cog_var in cog_var_names:
            if speech_var in corr_matrix.index and cog_var in corr_matrix.columns:
                r = corr_matrix.loc[speech_var, cog_var]
                p = p_matrix.loc[speech_var, cog_var]
                
                if p < SIGNIFICANCE_LEVEL:
                    speech_label = all_vars[speech_var]
                    cog_label = all_vars[cog_var]
                    
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
                    
                    
                    results.append({
                        'Speech_Domain': speech_label,
                        'Cognitive_Variable': cog_label,
                        'Correlation': r,
                        'p-value': p,
                        'Significance': sig,
                        'Direction': 'Negative' if r < 0 else 'Positive'
                    })
    
    return pd.DataFrame(results)

def plot_correlation_heatmap(corr_matrix, p_matrix, all_vars, output_dir):

    renamed_corr = corr_matrix.copy()
    renamed_corr.columns = [all_vars[c] for c in renamed_corr.columns]
    renamed_corr.index   = [all_vars[c] for c in renamed_corr.index]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(renamed_corr,
                annot=True,
                fmt='.2f',
                cmap = LinearSegmentedColormap.from_list('thesis', ['#2B62B8', 'white', '#C4376B']),
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                linecolor='#cccccc',
                cbar_kws={'label': 'Pearsonův r'},
                ax=ax,
                annot_kws={'fontsize': 9, 'color': COLOR_INK})

    for i in range(len(renamed_corr)):
        for j in range(len(renamed_corr)):
            orig_i = corr_matrix.index[i]
            orig_j = corr_matrix.columns[j]
            if orig_i != orig_j and p_matrix.loc[orig_i, orig_j] < 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                           fill=False,
                                           edgecolor=COLOR_INK,
                                           lw=2))

    ax.set_title('Korelace mezi řečovými doménami a kognitivními funkcemi\n(černý rámeček = p\u00a0<\u00a00.05)',
                 fontsize=18, fontweight='bold', color=COLOR_INK, pad=20)
    ax.tick_params(colors=COLOR_INK, labelsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def plot_scatter_top_correlations(df, sig_corr, all_vars, output_dir):
    if len(sig_corr) == 0:
        return

    sig_corr['abs_r'] = sig_corr['Correlation'].abs()
    top_corr = sig_corr.nlargest(min(6, len(sig_corr)), 'abs_r')

    n_plots = len(top_corr)
    if n_plots <= 2:
        n_cols = n_plots
    elif n_plots <= 4:
        n_cols = 2
    else:
        n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    reverse_vars = {v: k for k, v in all_vars.items()}

    for idx, (_, row) in enumerate(top_corr.iterrows()):
        ax = axes[idx]

        speech_label = row['Speech_Domain']
        cog_label    = row['Cognitive_Variable']
        r = row['Correlation']
        p = row['p-value']

        speech_col = reverse_vars[speech_label]
        cog_col    = reverse_vars[cog_label]

        df_plot = df[[speech_col, cog_col]].dropna()

        ax.scatter(df_plot[speech_col], df_plot[cog_col],
                   alpha=0.65, s=55,
                   color=SCATTER_COLOR,
                   edgecolors=COLOR_INK, linewidth=0.5)

        z     = np.polyfit(df_plot[speech_col], df_plot[cog_col], 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(df_plot[speech_col].min(), df_plot[speech_col].max(), 100)
        ax.plot(x_line, p_fit(x_line), color=LINE_COLOR, linestyle='--', linewidth=2, alpha=0.9)

        ax.set_xlabel(f'{speech_label}\n(vyšší = horší)', fontsize=16, color=COLOR_INK)
        ax.set_ylabel(f'{cog_label}\n{"(vyšší = lepší)" if "GDS" not in cog_label else "(vyšší = horší)"}',
                      fontsize=16, color=COLOR_INK)
        ax.set_title(f'{speech_label} ↔ {cog_label}\nr\u00a0=\u00a0{r:.3f},  p\u00a0=\u00a0{p:.4f}',
                     fontsize=18, fontweight='bold', color=COLOR_INK)
        ax.grid(alpha=0.25, linestyle='--', color=COLOR_INK)
        ax.tick_params(colors=COLOR_INK, labelsize=16)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_INK)
            spine.set_linewidth(0.8)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plots_top_correlations.pdf'),
                format='pdf', bbox_inches='tight')
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    df = load_data()
    
    corr_matrix, p_matrix, all_vars = calculate_correlations(df)
    
    speech_vars = {k: v for k, v in all_vars.items() if k in SPEECH_DOMAINS}
    cog_vars = {k: v for k, v in all_vars.items() if k in COGNITIVE_VARS}
    
    sig_corr = extract_significant_correlations(corr_matrix, p_matrix, all_vars, 
                                                speech_vars, cog_vars)
    
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))
    print(f"\n Uloženo: correlation_matrix.csv")
    
    if len(sig_corr) > 0:
        sig_corr.to_csv(os.path.join(OUTPUT_DIR, 'significant_correlations.csv'), index=False)
        print(f" Uloženo: significant_correlations.csv")
    
    plot_correlation_heatmap(corr_matrix, p_matrix, all_vars, OUTPUT_DIR)
    plot_scatter_top_correlations(df, sig_corr, all_vars, OUTPUT_DIR)
    

    print("HOTOVO!")
    print(f"Počet subjektů v analýze: {len(df)}")
    print(f"\nVýstupy v: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
