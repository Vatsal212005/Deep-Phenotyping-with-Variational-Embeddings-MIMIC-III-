# save as sanity_checks.py
import argparse
import os
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------- utils ----------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_float_cols(df: pd.DataFrame) -> pd.DataFrame:
    # keep only numeric columns (exclude string id cols like 'icustay_id')
    num = df.select_dtypes(include=[np.number]).copy()
    # drop all-NaN columns
    return num.loc[:, num.notna().any(axis=0)]

def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[WARN] matplotlib not available ({e}). Plots will be skipped. To enable: pip install matplotlib")
        return None

# ---------- checks ----------

def summarize_basic(df: pd.DataFrame) -> dict:
    out = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }
    return out

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss_cnt = df.isna().sum()
    miss_pct = miss_cnt / len(df) * 100.0
    summary = pd.DataFrame({"missing_count": miss_cnt, "missing_pct": miss_pct})
    summary = summary.sort_values("missing_pct", ascending=False)
    return summary

def find_constant_or_quasi(df: pd.DataFrame, quasi_threshold_unique_pct: float = 0.01) -> pd.DataFrame:
    n = len(df)
    uniq = df.nunique(dropna=False)
    uniq_pct = uniq / max(n, 1)
    const = pd.DataFrame({"unique_values": uniq, "unique_pct": uniq_pct})
    const["is_constant"] = const["unique_values"] <= 1
    const["is_quasi_const"] = (const["unique_pct"] <= quasi_threshold_unique_pct) & (~const["is_constant"])
    return const.sort_values(["is_constant", "is_quasi_const"], ascending=False)

def variance_rank(df_num: pd.DataFrame) -> pd.DataFrame:
    var = df_num.var(axis=0, ddof=0)
    return var.sort_values(ascending=False).rename("variance").to_frame()

def high_correlations(df_num: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    if df_num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_i", "feature_j", "corr"])
    corr = df_num.corr(method="pearson")
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c = corr.iat[i, j]
            if pd.notna(c) and abs(c) >= threshold:
                pairs.append((cols[i], cols[j], float(c)))
    return pd.DataFrame(pairs, columns=["feature_i", "feature_j", "corr"]).sort_values("corr", ascending=False)

def iqr_outlier_summary(df_num: pd.DataFrame, whisker: float = 1.5) -> pd.DataFrame:
    rows = []
    for col in df_num.columns:
        x = df_num[col].dropna()
        if x.empty:
            rows.append((col, 0, 0.0, np.nan, np.nan))
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lo = q1 - whisker * iqr
        hi = q3 + whisker * iqr
        out_cnt = int(((x < lo) | (x > hi)).sum())
        rows.append((col, out_cnt, out_cnt / len(x) * 100.0, lo, hi))
    return pd.DataFrame(rows, columns=["feature", "n_outliers", "outlier_pct", "low_cap", "high_cap"]).sort_values("outlier_pct", ascending=False)

# ---------- plots ----------

def plot_histograms(df_num: pd.DataFrame, outdir: str, max_cols: int = 24, bins: int = 30):
    plt = try_import_matplotlib()
    if plt is None:
        return
    cols = df_num.columns[:max_cols]
    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.axis("on")
        ax.hist(df_num[col].dropna().values, bins=bins)
        ax.set_title(col, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "histograms_top_numeric.png"), dpi=150)
    plt.close(fig)

def plot_corr_heatmap(df_num: pd.DataFrame, outdir: str, max_cols: int = 40):
    plt = try_import_matplotlib()
    if plt is None:
        return
    cols = df_num.columns[:max_cols]
    corr = df_num[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=6)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap (first {} numeric features)".format(len(cols)))
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "corr_heatmap.png"), dpi=150)
    plt.close(fig)

def plot_pca(df_num: pd.DataFrame, outdir: str):
    plt = try_import_matplotlib()
    if plt is None:
        return
    if df_num.shape[0] < 2 or df_num.shape[1] < 2:
        print("[WARN] Not enough data/columns for PCA plot; skipping.")
        return
    # simple standardization (mean-std) for PCA visualization only
    x = df_num.copy()
    x = (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if x.shape[0] < 2:
        print("[WARN] Not enough complete rows after NaN drop for PCA; skipping.")
        return
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(x.values)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z[:, 0], z[:, 1], s=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA (2D) — quick sanity check")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_scatter.png"), dpi=150)
    plt.close(fig)

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Step 1: sanity checks for features_clean.csv")
    parser.add_argument("--features", required=True, help="Path to features_clean.csv")
    parser.add_argument("--outdir", required=True, help="Folder to write sanity-check reports/plots")
    parser.add_argument("--id-col", default="icustay_id", help="Identifier column name (default: icustay_id)")
    parser.add_argument("--corr-threshold", type=float, default=0.95, help="Absolute correlation threshold")
    parser.add_argument("--quasi-unique-pct", type=float, default=0.01, help="Quasi-constant unique% threshold (0.01 = 1%)")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    print(f"[INFO] Loading {args.features}")
    df = pd.read_csv(args.features)
    # move id to front if present
    cols = list(df.columns)
    if args.id_col in cols:
        cols = [args.id_col] + [c for c in cols if c != args.id_col]
        df = df[cols]

    # --- summaries ---
    basic = summarize_basic(df)
    miss = summarize_missing(df)
    const = find_constant_or_quasi(df, quasi_threshold_unique_pct=args.quasi_unique_pct)

    # numeric-only for stats/plots
    df_num = safe_float_cols(df.drop(columns=[args.id_col], errors="ignore"))

    var_rank = variance_rank(df_num)
    high_corr = high_correlations(df_num, threshold=args.corr_threshold)
    outliers = iqr_outlier_summary(df_num)

    # --- save artifacts ---
    pd.Series(basic).to_json(os.path.join(args.outdir, "basic_summary.json"), indent=2)
    miss.to_csv(os.path.join(args.outdir, "missingness_summary.csv"))
    const.to_csv(os.path.join(args.outdir, "constant_quasi_constant.csv"))
    var_rank.to_csv(os.path.join(args.outdir, "variance_rank.csv"))
    high_corr.to_csv(os.path.join(args.outdir, "high_correlations.csv"), index=False)
    outliers.to_csv(os.path.join(args.outdir, "iqr_outliers.csv"), index=False)

    # --- plots (best-effort) ---
    plot_histograms(df_num, args.outdir, max_cols=24, bins=30)
    plot_corr_heatmap(df_num, args.outdir, max_cols=40)
    plot_pca(df_num, args.outdir)

    # --- console highlights ---
    print(f"[INFO] Rows × Cols: {basic['n_rows']} × {basic['n_cols']}")
    print("[INFO] Top 10 missing columns:")
    print(miss.head(10))
    print("[INFO] Top 10 variance columns:")
    print(var_rank.head(10))
    if not high_corr.empty:
        print(f"[INFO] High-corr pairs (|r|≥{args.corr_threshold}): {len(high_corr)} (see high_correlations.csv)")
    else:
        print("[INFO] No highly correlated pairs found at the given threshold.")
    print("[INFO] Reports saved to:", args.outdir)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
