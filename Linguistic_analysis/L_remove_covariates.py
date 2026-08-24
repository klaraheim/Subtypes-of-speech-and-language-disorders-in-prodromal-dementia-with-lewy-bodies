# Odstranění kovariátů lingvistické části dat

import os
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LINGUISTIC_PATH = os.path.join(SCRIPT_DIR, "Linguistic_results.xlsx")
METADATA_PATH = os.path.join(SCRIPT_DIR, "metadata_altered_plusMedian.xlsx")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_linguistic_residualised")

COVARIATE_COLS = ["age", "gender", "education length"]
GROUP_CANDIDATES = ["Category", "category", "disease", "Disease", "Group", "group", "Diagnosis"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def find_group_column(df: pd.DataFrame):
    for col in GROUP_CANDIDATES:
        if col in df.columns:
            return col
    return None

def coerce_gender_to_numeric(series: pd.Series) -> pd.Series:

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "m": 1, "male": 1, "man": 1, "1": 1, "true": 1,
        "f": 0, "female": 0, "woman": 0, "0": 0, "false": 0
    }
    return s.map(mapping).astype(float)

def load_and_merge(linguistic_path: str, metadata_path: str) -> pd.DataFrame:
    if not os.path.exists(linguistic_path):
        raise FileNotFoundError(f"Soubor nenalezen: {linguistic_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Soubor nenalezen: {metadata_path}")

    linguistic_df = pd.read_excel(linguistic_path, engine="openpyxl")
    metadata_df = pd.read_excel(metadata_path, engine="openpyxl")

    if "Subject" not in linguistic_df.columns or "Subject" not in metadata_df.columns:
        raise ValueError("Chybí sloupec 'Subject' v linguistic nebo metadata souboru.")

    merged = linguistic_df.merge(metadata_df, on="Subject", how="inner", suffixes=("", "_meta"))
    return merged

def select_linguistic_feature_cols(merged: pd.DataFrame):
 
    linguistic_prefixes = ("Lex_", "Sem_", "Syn_")
    feature_cols = [c for c in merged.columns if c.startswith(linguistic_prefixes)]

    if len(feature_cols) == 0:
        raise ValueError("Nenalezeny žádné lingvistické featury s prefixy Lex_/Sem_/Syn_.")

    return feature_cols

def prepare_df(merged: pd.DataFrame, feature_cols, cov_cols, group_col) -> pd.DataFrame:
    keep_cols = ["Subject"] + ([group_col] if group_col else []) + cov_cols + feature_cols
    df = merged[keep_cols].copy()

    if "gender" in df.columns:
        df["gender"] = coerce_gender_to_numeric(df["gender"])

    for c in cov_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for f in feature_cols:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Subject"] + cov_cols + feature_cols)
    after = len(df)

    if after < before:
        pass

    return df

def design_matrix(df: pd.DataFrame, cov_cols):
    X_cov = df[cov_cols].to_numpy(dtype=float)
    intercept = np.ones((X_cov.shape[0], 1), dtype=float)
    return np.hstack([intercept, X_cov])

def residualize(df: pd.DataFrame, feature_cols, cov_cols):
    X = design_matrix(df, cov_cols)
    Y = df[feature_cols].to_numpy(dtype=float)

    beta, *_ = lstsq(X, Y, rcond=None)
    Y_hat = X @ beta
    Y_res = Y - Y_hat

    out = df.copy()
    out[feature_cols] = out[feature_cols].astype(float)
    out.loc[:, feature_cols] = Y_res
    return out, beta

def pearson_corr_matrix(df: pd.DataFrame, feature_cols, cov_cols) -> pd.DataFrame:
    corr = np.zeros((len(feature_cols), len(cov_cols)))

    for i, f in enumerate(feature_cols):
        y = df[f].to_numpy(dtype=float)
        for j, c in enumerate(cov_cols):
            x = df[c].to_numpy(dtype=float)

            if np.std(x) == 0 or np.std(y) == 0:
                r = np.nan
            else:
                r = np.corrcoef(x, y)[0, 1]

            corr[i, j] = r

    return pd.DataFrame(corr, index=feature_cols, columns=cov_cols)

def plot_heatmap(corr_df, title, filename):
    plt.figure(figsize=(10, min(0.35 * len(corr_df), 20)))
    sns.heatmap(
        corr_df,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        linecolor="gray",
        cbar_kws={"label": "Pearson r"}
    )
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Kovariáty", fontsize=16)
    plt.ylabel("Lingvistické parametry", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def plot_histogram(corr_df, title, filename):
    abs_corr = np.abs(corr_df.values.flatten())
    abs_corr = abs_corr[~np.isnan(abs_corr)]

    plt.figure(figsize=(8, 6))
    plt.hist(abs_corr, bins=30, edgecolor="black", alpha=0.75)
    plt.xlabel("|Pearson r|", fontsize=16)
    plt.ylabel("Četnost", fontsize=16)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.tick_params(labelsize=16)
    plt.axvline(0.3, linestyle="--", label="|r| = 0.3")
    plt.axvline(0.5, linestyle="--", label="|r| = 0.5")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def save_outputs(df_before, df_after, beta, feature_cols, cov_cols, group_col):
    ensure_dir(OUTPUT_DIR)

    out_cols = ["Subject"] + ([group_col] if group_col else []) + feature_cols
    df_after[out_cols].to_csv(
        os.path.join(OUTPUT_DIR, "linguistic_residualized.csv"),
        index=False
    )

    predictors = ["Intercept"] + cov_cols
    model_df = pd.DataFrame(beta, index=predictors, columns=feature_cols)
    model_df.to_csv(
        os.path.join(OUTPUT_DIR, "linguistic_residualization_model.csv"),
        index=True
    )

    corr_before = pearson_corr_matrix(df_before, feature_cols, cov_cols)
    corr_after = pearson_corr_matrix(df_after, feature_cols, cov_cols)

    corr_before.to_csv(os.path.join(OUTPUT_DIR, "corr_before_pearson.csv"))
    corr_after.to_csv(os.path.join(OUTPUT_DIR, "corr_after_pearson.csv"))

    plot_heatmap(
        corr_before,
        "Pearson korelace PŘED odstraněním kovariátů",
        "pearson_before_heatmap.png"
    )

    plot_heatmap(
        corr_after,
        "Pearson korelace PO odstranění kovariátů",
        "pearson_after_heatmap.png"
    )

    plot_histogram(
        corr_before,
        "Distribuce |r| PŘED odstraněním kovariátů",
        "pearson_before_hist.png"
    )

    plot_histogram(
        corr_after,
        "Distribuce |r| PO odstranění kovariátů",
        "pearson_after_hist.png"
    )

    print("\n Uloženo do:", OUTPUT_DIR)

    print(f"  průměrná |r| PŘED: {np.nanmean(np.abs(corr_before.values)):.3f}")
    print(f"  průměrná |r| PO:   {np.nanmean(np.abs(corr_after.values)):.3f}")
    print(f"  max |r| PŘED:      {np.nanmax(np.abs(corr_before.values)):.3f}")
    print(f"  max |r| PO:        {np.nanmax(np.abs(corr_after.values)):.3f}")

def main():

    merged = load_and_merge(LINGUISTIC_PATH, METADATA_PATH)

    group_col = find_group_column(merged)

    cov_cols = [c for c in COVARIATE_COLS if c in merged.columns]
    if len(cov_cols) != len(COVARIATE_COLS):
        missing = [c for c in COVARIATE_COLS if c not in merged.columns]
        raise ValueError(f"Chybí kovariáty v merged datech: {missing}")

    feature_cols = select_linguistic_feature_cols(merged)

    df = prepare_df(merged, feature_cols, cov_cols, group_col)

    df_res, beta = residualize(df, feature_cols, cov_cols)

    save_outputs(df, df_res, beta, feature_cols, cov_cols, group_col)

if __name__ == "__main__":
    main()