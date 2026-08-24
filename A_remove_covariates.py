# Odstranění kovariátů akustické části dat

import os
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ACOUSTIC_PATH = "Acoustic_results.xlsx"
METADATA_PATH = "metadata_altered_plusMedian.xlsx"
OUTPUT_DIR = "output_acoustic_residualised"

COVARIATE_COLS = ["age", "gender", "education length"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def coerce_gender_to_numeric(series: pd.Series) -> pd.Series:

    
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "m": 1, "male": 1, "man": 1, "1": 1, "true": 1,
        "f": 0, "female": 0, "woman": 0, "0": 0, "false": 0
    }
    return s.map(mapping).astype(float)

def load_and_merge(acoustic_path: str, metadata_path: str) -> pd.DataFrame:
    if not os.path.exists(acoustic_path):
        raise FileNotFoundError(f"Soubor nenalezen: {acoustic_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Soubor nenalezen: {metadata_path}")

    acoustic_df = pd.read_excel(acoustic_path, engine="openpyxl")
    metadata_df = pd.read_excel(metadata_path, engine="openpyxl")

    if "Subject" not in acoustic_df.columns or "Subject" not in metadata_df.columns:
        raise ValueError("Chybí sloupec 'Subject' v acoustic nebo metadata souboru.")

    merged = acoustic_df.merge(metadata_df, on="Subject", how="inner", suffixes=("", "_meta"))
    return merged

def select_feature_cols(merged: pd.DataFrame, cov_cols):
    id_cols = ["Subject"]
    if "Category" in merged.columns:
        id_cols.append("Category")

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

    exclude = set(id_cols + cov_cols)

    feature_cols = [c for c in numeric_cols if c not in exclude]

    if len(feature_cols) == 0:
        raise ValueError("Nenalezeny žádné numerické akustické featury.")

    return feature_cols, id_cols

def prepare_df(merged: pd.DataFrame, feature_cols, cov_cols, id_cols) -> pd.DataFrame:
    df = merged[id_cols + cov_cols + feature_cols].copy()

    if "gender" in df.columns:
        df["gender"] = coerce_gender_to_numeric(df["gender"])

    for c in cov_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for f in feature_cols:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    before = len(df)
    df = df.dropna(subset=cov_cols + feature_cols + ["Subject"])
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
            r = np.corrcoef(x, y)[0, 1]
            corr[i, j] = r
    return pd.DataFrame(corr, index=feature_cols, columns=cov_cols)

def plot_heatmap(corr_df, title, filename):
    plt.figure(figsize=(10, min(0.4 * len(corr_df), 20)))
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
    plt.ylabel("Akustické parametry", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def plot_histogram(corr_df, title, filename):
    abs_corr = np.abs(corr_df.values.flatten())

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

def save_outputs(df_before, df_after, beta, feature_cols, cov_cols, id_cols):
    ensure_dir(OUTPUT_DIR)

    out_cols = ["Subject"] + (["Category"] if "Category" in df_after.columns else []) + feature_cols
    df_after[out_cols].to_csv(os.path.join(OUTPUT_DIR, "acoustic_residualized.csv"), index=False)

    predictors = ["Intercept"] + cov_cols
    model_df = pd.DataFrame(beta, index=predictors, columns=feature_cols)
    model_df.to_csv(os.path.join(OUTPUT_DIR, "acoustic_residualization_model.csv"), index=True)

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

    print(f"  průměrná |r| PŘED: {np.mean(np.abs(corr_before.values)):.3f}")
    print(f"  průměrná |r| PO:   {np.mean(np.abs(corr_after.values)):.3f}")
    print(f"  max |r| PŘED:      {np.max(np.abs(corr_before.values)):.3f}")
    print(f"  max |r| PO:        {np.max(np.abs(corr_after.values)):.3f}")

def main():

    merged = load_and_merge(ACOUSTIC_PATH, METADATA_PATH)

    cov_cols = [c for c in COVARIATE_COLS if c in merged.columns]
    if len(cov_cols) != len(COVARIATE_COLS):
        missing = [c for c in COVARIATE_COLS if c not in merged.columns]
        raise ValueError(f"Chybí kovariáty v merged datech: {missing}")

    feature_cols, id_cols = select_feature_cols(merged, cov_cols)

    df = prepare_df(merged, feature_cols, cov_cols, id_cols)

    df_res, beta = residualize(df, feature_cols, cov_cols)

    save_outputs(df, df_res, beta, feature_cols, cov_cols, id_cols)

if __name__ == "__main__":
    main()
