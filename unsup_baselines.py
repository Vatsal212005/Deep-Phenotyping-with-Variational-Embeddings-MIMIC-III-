# save as unsup_baselines.py
import argparse
import os
import sys
import json
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

# plotting optional
def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[WARN] matplotlib not available ({e}). Plots will be skipped. To enable: pip install matplotlib")
        return None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any(axis=0)]

def load_features_or_embeddings(
    features_path: str,
    id_col: str,
    use_embeddings: bool,
) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(features_path)
    if use_embeddings:
        # Expect embeddings.csv with columns: id_col, z1..zk
        if id_col not in df.columns:
            # Try common first column
            first = df.columns[0]
            print(f"[WARN] '{id_col}' not found in embeddings; using first column '{first}' as ID.")
            id_col = first
        ids = df[id_col]
        X_df = safe_numeric_df(df.drop(columns=[id_col], errors="ignore"))
        return ids, X_df
    else:
        ids = df[id_col] if id_col in df.columns else pd.Series(np.arange(len(df)), name=id_col)
        X_df = safe_numeric_df(df.drop(columns=[id_col], errors="ignore"))
        return ids, X_df

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------- KMEANS --------------------

def cmd_kmeans(args):
    outdir = ensure_dir(os.path.join(args.outdir, "kmeans_embeddings" if args.use_embeddings else "kmeans_features"))
    ids, X_df = load_features_or_embeddings(args.features, args.id_col, args.use_embeddings)
    if X_df.shape[0] < 2 or X_df.shape[1] < 1:
        print("[ERROR] Not enough data for KMeans.")
        return

    # Standardize for clustering stability (even if features_clean already robust-scaled)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(np.float32))
    pd.DataFrame({"feature": X_df.columns, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(
        os.path.join(outdir, "standardizer_params.csv"), index=False
    )

    # Determine K
    k_list = list(range(max(2, args.k_min), max(args.k_max, args.k_min)+1))
    elbow = []
    sils = []
    for k in k_list:
        km = KMeans(n_clusters=k, n_init="auto", random_state=args.seed)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        elbow.append((k, inertia))
        s = silhouette_score(X, labels) if X.shape[0] > k else np.nan
        sils.append((k, float(s)))

    elbow_df = pd.DataFrame(elbow, columns=["k", "inertia"])
    sil_df = pd.DataFrame(sils, columns=["k", "silhouette"])
    elbow_df.to_csv(os.path.join(outdir, "elbow.csv"), index=False)
    sil_df.to_csv(os.path.join(outdir, "silhouette.csv"), index=False)

    # Auto-pick K = argmax silhouette unless user provides --k
    if args.k is not None and args.k >= 2:
        best_k = args.k
    else:
        sel = sil_df.dropna()
        best_k = int(sel.loc[sel["silhouette"].idxmax(), "k"]) if not sel.empty else 2

    print(f"[INFO] Selected K={best_k}")

    km = KMeans(n_clusters=best_k, n_init="auto", random_state=args.seed)
    labels = km.fit_predict(X)

    # Save assignments
    assign = pd.DataFrame({args.id_col: ids.values, "cluster": labels})
    assign.to_csv(os.path.join(outdir, "cluster_assignments.csv"), index=False)

    # Save cluster sizes
    sizes = assign["cluster"].value_counts().sort_index()
    sizes.to_csv(os.path.join(outdir, "cluster_sizes.csv"), header=["count"])

    # Save centroids in original feature space (approx invert scaling)
    centroids_std = km.cluster_centers_
    centroids = centroids_std * scaler.scale_ + scaler.mean_
    cent_df = pd.DataFrame(centroids, columns=X_df.columns)
    cent_df.insert(0, "cluster", np.arange(best_k))
    cent_df.to_csv(os.path.join(outdir, "cluster_centroids.csv"), index=False)

    # Plots
    plt = try_import_matplotlib()
    if plt is not None:
        # elbow
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
        ax.set_title("Elbow (inertia vs k)")
        ax.set_xlabel("k")
        ax.set_ylabel("inertia")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "elbow.png"), dpi=150)
        plt.close(fig)

        # silhouette
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(sil_df["k"], sil_df["silhouette"], marker="o")
        ax.set_title("Silhouette vs k")
        ax.set_xlabel("k")
        ax.set_ylabel("silhouette")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "silhouette.png"), dpi=150)
        plt.close(fig)

    print(f"[DONE] KMeans artifacts saved to: {outdir}")

# -------------------- ISOLATION FOREST --------------------

def cmd_isoforest(args):
    outdir = ensure_dir(os.path.join(args.outdir, "isoforest_embeddings" if args.use_embeddings else "isoforest_features"))
    ids, X_df = load_features_or_embeddings(args.features, args.id_col, args.use_embeddings)
    if X_df.shape[0] < 2 or X_df.shape[1] < 1:
        print("[ERROR] Not enough data for IsolationForest.")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(np.float32))

    iso = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.seed,
        bootstrap=True,
        n_jobs=-1,
    )
    iso.fit(X)
    scores = -iso.score_samples(X)  # higher => more anomalous
    preds = iso.predict(X)  # -1 outlier, +1 inlier

    out = pd.DataFrame({
        args.id_col: ids.values,
        "anomaly_score": scores,
        "is_outlier": (preds == -1).astype(int)
    }).sort_values("anomaly_score", ascending=False)
    out.to_csv(os.path.join(outdir, "anomaly_scores.csv"), index=False)

    topn = int(args.top_n) if args.top_n > 0 else min(20, len(out))
    out.head(topn).to_csv(os.path.join(outdir, f"top_{topn}_anomalies.csv"), index=False)

    # Plot distribution
    plt = try_import_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(scores, bins=30)
        ax.set_title("IsolationForest anomaly score distribution")
        ax.set_xlabel("score (higher = more anomalous)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "anomaly_score_hist.png"), dpi=150)
        plt.close(fig)

    print(f"[DONE] IsolationForest artifacts saved to: {outdir}")

# -------------------- LOCAL OUTLIER FACTOR --------------------

def cmd_lof(args):
    outdir = ensure_dir(os.path.join(args.outdir, "lof_embeddings" if args.use_embeddings else "lof_features"))
    ids, X_df = load_features_or_embeddings(args.features, args.id_col, args.use_embeddings)
    if X_df.shape[0] < 2 or X_df.shape[1] < 1:
        print("[ERROR] Not enough data for LOF.")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values.astype(np.float32))

    lof = LocalOutlierFactor(
        n_neighbors=args.n_neighbors,
        contamination=args.contamination if args.contamination > 0 else "auto",
        novelty=False,  # classic LOF (no separate predict on unseen)
        n_jobs=-1
    )
    preds = lof.fit_predict(X)  # -1 outlier, +1 inlier
    scores = -lof.negative_outlier_factor_  # higher => more anomalous

    out = pd.DataFrame({
        args.id_col: ids.values,
        "anomaly_score": scores,
        "is_outlier": (preds == -1).astype(int)
    }).sort_values("anomaly_score", ascending=False)
    out.to_csv(os.path.join(outdir, "lof_scores.csv"), index=False)

    topn = int(args.top_n) if args.top_n > 0 else min(20, len(out))
    out.head(topn).to_csv(os.path.join(outdir, f"top_{topn}_lof_anomalies.csv"), index=False)

    plt = try_import_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(scores, bins=30)
        ax.set_title("LOF anomaly score distribution")
        ax.set_xlabel("score (higher = more anomalous)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "lof_score_hist.png"), dpi=150)
        plt.close(fig)

    print(f"[DONE] LOF artifacts saved to: {outdir}")

# -------------------- ANALYZE --------------------

def cmd_analyze(args):
    outdir = ensure_dir(os.path.join(args.outdir, "analysis"))
    ids, X_df = load_features_or_embeddings(args.features, args.id_col, args.use_embeddings)
    X = X_df.values.astype(np.float32)

    report_lines: List[str] = []
    report_lines.append(f"Rows: {X_df.shape[0]}\nCols: {X_df.shape[1]}\n")

    # If cluster file provided, compute per-cluster stats
    if args.cluster_csv and os.path.exists(args.cluster_csv):
        cl = pd.read_csv(args.cluster_csv)
        if args.id_col not in cl.columns:
            print(f"[WARN] '{args.id_col}' not found in cluster CSV; attempting to align on first column.")
            cl.rename(columns={cl.columns[0]: args.id_col}, inplace=True)
        merged = pd.DataFrame({args.id_col: ids.values}).merge(cl[[args.id_col, "cluster"]], on=args.id_col, how="left")
        if "cluster" in merged.columns:
            report_lines.append("Per-cluster top differentiating features (mean diff vs global):\n")
            global_mean = X_df.mean(axis=0)
            out_rows = []
            for k, sub in merged.groupby("cluster"):
                mask = sub["cluster"].notna() & (sub["cluster"] == k)
                idx = mask.values
                if idx.sum() < 2:
                    continue
                mean_k = X_df.loc[idx].mean(axis=0)
                diff = (mean_k - global_mean).abs().sort_values(ascending=False)
                top = diff.head(min(15, len(diff)))
                for feat, val in top.items():
                    out_rows.append((k, feat, float(val)))
            diff_df = pd.DataFrame(out_rows, columns=["cluster", "feature", "abs_mean_diff"]).sort_values(["cluster","abs_mean_diff"], ascending=[True, False])
            diff_df.to_csv(os.path.join(outdir, "cluster_top_features.csv"), index=False)
            report_lines.append("Saved: cluster_top_features.csv\n")

    # If anomaly score file provided, correlate with features
    if args.anomaly_csv and os.path.exists(args.anomaly_csv):
        an = pd.read_csv(args.anomaly_csv)
        if args.id_col not in an.columns:
            an.rename(columns={an.columns[0]: args.id_col}, inplace=True)
        if "anomaly_score" in an.columns:
            merged = pd.DataFrame({args.id_col: ids.values}).merge(an[[args.id_col, "anomaly_score"]], on=args.id_col, how="left")
            scores = merged["anomaly_score"].values.astype(np.float32)
            corrs = []
            for col in X_df.columns:
                x = X_df[col].values.astype(np.float32)
                if np.std(x) < 1e-8:
                    continue
                c = np.corrcoef(x, scores)[0,1]
                if np.isfinite(c):
                    corrs.append((col, float(c), float(abs(c))))
            corr_df = pd.DataFrame(corrs, columns=["feature","pearson_corr","abs_corr"]).sort_values("abs_corr", ascending=False)
            corr_df.to_csv(os.path.join(outdir, "anomaly_score_feature_correlation.csv"), index=False)
            report_lines.append("Saved: anomaly_score_feature_correlation.csv\n")

    # Write a mini README/report
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("Mini Evaluation Report\n")
        f.write("=====================\n\n")
        for line in report_lines:
            f.write(line)

    print(f"[DONE] Analysis artifacts saved to: {outdir}")

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Unsupervised baselines: KMeans, IsolationForest, LOF, and analyze helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # shared args maker
    def add_shared(p):
        p.add_argument("--features", required=True, help="Path to features_clean.csv or embeddings.csv")
        p.add_argument("--outdir", required=True, help="Output directory (subfolders will be created)")
        p.add_argument("--id-col", default="icustay_id")
        p.add_argument("--use-embeddings", action="store_true", help="Treat --features as embeddings.csv")

    # kmeans
    p_km = sub.add_parser("kmeans", help="KMeans clustering")
    add_shared(p_km)
    p_km.add_argument("--k", type=int, default=None, help="If set, fix K (>=2); otherwise, auto-pick by max silhouette")
    p_km.add_argument("--k-min", type=int, default=2)
    p_km.add_argument("--k-max", type=int, default=10)
    p_km.add_argument("--seed", type=int, default=42)
    p_km.set_defaults(func=cmd_kmeans)

    # isoforest
    p_iso = sub.add_parser("isoforest", help="IsolationForest anomaly detection")
    add_shared(p_iso)
    p_iso.add_argument("--n-estimators", type=int, default=200)
    p_iso.add_argument("--contamination", type=float, default=0.10, help="Approx outlier fraction (0..0.5)")
    p_iso.add_argument("--top-n", type=int, default=20)
    p_iso.add_argument("--seed", type=int, default=42)
    p_iso.set_defaults(func=cmd_isoforest)

    # lof
    p_lof = sub.add_parser("lof", help="Local Outlier Factor (optional)")
    add_shared(p_lof)
    p_lof.add_argument("--n-neighbors", type=int, default=20)
    p_lof.add_argument("--contamination", type=float, default=0.10)
    p_lof.add_argument("--top-n", type=int, default=20)
    p_lof.set_defaults(func=cmd_lof)

    # analyze
    p_an = sub.add_parser("analyze", help="Light reporting for clusters/anomalies")
    add_shared(p_an)
    p_an.add_argument("--cluster-csv", default="", help="Path to cluster_assignments.csv")
    p_an.add_argument("--anomaly-csv", default="", help="Path to anomaly_scores.csv or lof_scores.csv")
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
