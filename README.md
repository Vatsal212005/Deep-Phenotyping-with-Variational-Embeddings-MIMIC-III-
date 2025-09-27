# Latent Clinical Intelligence: Deep Unsupervised Phenotyping with VAE (MIMIC-III)

> A high-reliability unsupervised ML pipeline for ICU data: **robust preprocessing → Variational Autoencoder embeddings → KMeans clustering → Isolation Forest / LOF anomalies → light reporting & plots**.  
> Demonstrated on **MIMIC-III demo v1.4** (first 24h of ICU); methods scale to full MIMIC-III/IV.

---

## 🔥 Highlights

- **End-to-end pipeline** on real ICU data (CHARTEVENTS/LABEVENTS → 24h features → models → artifacts).  
- **VAE (β-VAE)** learns compact **patient embeddings** + **reconstruction error** (outlier signal).  
- **Unsupervised baselines**: KMeans (elbow/silhouette), IsolationForest (and LOF) with ranked anomalies.  
- **Interpretability hooks**: cluster-wise top features, anomaly-feature correlations, concise report.  
- **Windows-first** (PowerShell), Python 3.12, **minimal deps**, optional UMAP/matplotlib guarded with graceful fallbacks.  
- **Demo results (from a real run)**:  
  - Silhouette (embeddings): **K=2 → 0.781; K=3 → 0.779** (strong structure)  
  - K=2 sizes: **126 vs 3** (tiny “rare” subgroup)  
  - Isolation Forest flagged **~10%** outliers (top scores ~0.75)

---

## 📌 Repository Structure

```
.
├─ mimic_preprocess.py        # Build first-24h ICU feature matrix (robust, unit hygiene, scaling)
├─ sanity_checks.py           # QC: missingness, variance, correlations, IQR outliers, PCA/histos/heatmap
├─ vae_train.py               # Train β-VAE (or infer): embeddings.csv, recon_error.csv, plots, model
├─ unsup_baselines.py         # Subcommands: kmeans / isoforest / lof / analyze
├─ requirements.txt           # Reproducible environment (add torch variant as needed)
├─ .gitignore                 # Keeps large data/models out of Git
└─ README.md                  # You are here
```
> **Not versioned** (by design): `Dataset/`, `mimic_outputs/`, `mimic_reports/`, `models/` (see `.gitignore`).

---

## 🧠 Project Motivation

Clinical ICU data is high-dimensional, noisy, and often unlabeled in early windows.  
This project **discovers structure without labels**:

- **Phenotype discovery**: group patients with similar first-24h profiles (e.g., renal vs sepsis-like signatures).  
- **Outlier detection**: surface rare/atypical cases (potentially critical, or data quality issues).  

> This is a **methods/engineering** project: robust preprocessing + unsupervised representation + clustering/anomalies + interpretability.  
> For clinical claims, scale to full MIMIC and validate against outcomes.

---

## 🧰 Environment & Setup (Windows, PowerShell)

> Python 3.12, venv recommended.

```powershell
# 0) Create & activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip

# 1) Core deps
pip install -r requirements.txt

# 2a) CPU-only PyTorch (simple demo is fine on CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2b) GPU (RTX 40xx / CUDA 12.4 wheels) — preferred if you have a 4060:
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) Optional plotting & 2D projection
pip install matplotlib umap-learn
```

**Verify GPU** (optional):

```powershell
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('device:', torch.cuda.get_device_name(0))
PY
```

---

## 🗂️ Dataset

- **MIMIC-III clinical database demo v1.4** (Kaggle mirror, same schema as full MIMIC-III).  
- Local path (example):

```
C:\Users\ironm\OneDrive\Desktop\Resume Project\Dataset\mimic-iii-clinical-database-demo-1.4
```

> Demo subset is perfect for pipeline engineering. For clinical generalization, run the same code on full MIMIC-III/IV.

---

## 🧱 Pipeline Overview
```markdown
```mermaid
flowchart LR
    A[Raw MIMIC-III tables: CHARTEVENTS, LABEVENTS, D_*] --> B[Preprocess: mimic_preprocess.py (24h window, scaling)]
    B --> C[Sanity Checks: sanity_checks.py (missingness, variance, correlations, PCA)]
    C --> D[VAE Training: vae_train.py (beta-VAE, early stop, embeddings.csv)]
    D --> E[KMeans: unsup_baselines.py kmeans (elbow, silhouette, centroids)]
    D --> F[IsoForest/LOF: unsup_baselines.py isoforest/lof (anomaly_scores.csv)]
    E --> G[Analyze: unsup_baselines.py analyze (cluster features)]
    F --> G
```
---

## 🚀 How to Run (copy-paste commands)

**Paths used below:**

- Features:  
  `C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_outputs\features_clean.csv`  
- Artifacts outdir:  
  `C:\Users\ironm\OneDrive\Desktop\Resume Project\models`

### 1) (If needed) Build features
```powershell
python mimic_preprocess.py `
  --dataset "C:\Users\ironm\OneDrive\Desktop\Resume Project\Dataset\mimic-iii-clinical-database-demo-1.4" `
  --outdir  "C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_outputs"
```

### 2) Sanity checks
```powershell
python sanity_checks.py `
  --features "C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_outputs\features_clean.csv" `
  --outdir   "C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_reports" `
  --id-col icustay_id
```

### 3) Train β-VAE → embeddings + recon error
```powershell
python vae_train.py `
  --features "C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_outputs\features_clean.csv" `
  --outdir   "C:\Users\ironm\OneDrive\Desktop\Resume Project\models" `
  --id-col icustay_id `
  --hidden-sizes "256,128" `
  --latent-dim 8 `
  --beta 1.0 `
  --lr 1e-3 `
  --batch-size 32 `
  --epochs 200 `
  --weight-decay 1e-5 `
  --val-split 0.2 `
  --patience 20 `
  --seed 42 `
  --standardize true
```

Outputs → `models/vae/`:  
`vae_model.pt`, `vae_config.json`, `scaler.joblib` (if used), `embeddings.csv`, `recon_error.csv`, `loss_curves.png`, `recon_error_hist.png`, `embeddings_2d.png`

### 4) KMeans on embeddings
```powershell
python unsup_baselines.py kmeans `
  --features "C:\Users\ironm\OneDrive\Desktop\Resume Project\models\vae\embeddings.csv" `
  --use-embeddings `
  --outdir   "C:\Users\ironm\OneDrive\Desktop\Resume Project\models" `
  --id-col icustay_id `
  --k-min 2 --k-max 6 `
  --seed 42
```

Outputs → `models/kmeans_embeddings/`:  
`cluster_assignments.csv`, `cluster_sizes.csv`, `cluster_centroids.csv`, `elbow.csv/png`, `silhouette.csv/png`

### 5) Isolation Forest (anomalies) on embeddings
```powershell
python unsup_baselines.py isoforest `
  --features "C:\Users\ironm\OneDrive\Desktop\Resume Project\models\vae\embeddings.csv" `
  --use-embeddings `
  --outdir   "C:\Users\ironm\OneDrive\Desktop\Resume Project\models" `
  --id-col icustay_id `
  --n-estimators 300 `
  --contamination 0.10 `
  --top-n 20 `
  --seed 42
```

Outputs → `models/isoforest_embeddings/`:  
`anomaly_scores.csv`, `top_20_anomalies.csv`, `anomaly_score_hist.png`

### 6) (Optional) LOF
```powershell
python unsup_baselines.py lof `
  --features "C:\Users\ironm\OneDrive\Desktop\Resume Project\models\vae\embeddings.csv" `
  --use-embeddings `
  --outdir   "C:\Users\ironm\OneDrive\Desktop\Resume Project\models" `
  --id-col icustay_id `
  --n-neighbors 15 `
  --contamination 0.10 `
  --top-n 20
```

### 7) Interpret clusters & anomalies
```powershell
python unsup_baselines.py analyze `
  --features    "C:\Users\ironm\OneDrive\Desktop\Resume Project\mimic_outputs\features_clean.csv" `
  --outdir      "C:\Users\ironm\OneDrive\Desktop\Resume Project\models" `
  --id-col icustay_id `
  --cluster-csv "C:\Users\ironm\OneDrive\Desktop\Resume Project\models\kmeans_embeddings\cluster_assignments.csv" `
  --anomaly-csv "C:\Users\ironm\OneDrive\Desktop\Resume Project\models\isoforest_embeddings\anomaly_scores.csv"
```

Outputs → `models/analysis/`:  
`cluster_top_features.csv`, `anomaly_score_feature_correlation.csv`, `report.txt`

---

## 🧩 What Each Script Does

### `mimic_preprocess.py`
- Normalizes mixed-case column names.
- 24h window aggregation by ICU stay; unit hygiene (e.g., °F→°C), physiologic clipping.
- Regex extracts labs/vitals via `D_ITEMS` and `D_LABITEMS`.
- Hourly aggregation → summary stats (mean/min/max/std/coverage), **median imputation**, **missingness indicators**, **RobustScaler**.
- Outputs:  
  - `cohort.(csv|parquet)`  
  - `features_raw.(csv|parquet)`  
  - `features_clean.(csv|parquet)`  
  - `imputer_median.joblib`, `scaler_robust.joblib`, `feature_metadata.json`

### `sanity_checks.py`
- Basic summary, missingness, constant/quasi-constant, variance rank.
- Pearson high-correlation pairs, IQR outlier counts.
- Plots (best-effort): histograms, corr heatmap, PCA(2D).
- Outputs CSVs + PNGs to a given `--outdir`.

### `vae_train.py`
- **β-VAE** with MLP encoder/decoder, MSE recon + β·KL.  
- Train/val split with early stopping (`--val-split`, `--patience`).
- Optional StandardScaler post-robust scaling (`--standardize`).
- Saves model/config/scaler, **embeddings.csv** (μ), **recon_error.csv**, and plots.
- `--infer` mode: load `vae_model.pt` for any features file and re-export embeddings/errors.

### `unsup_baselines.py`
- `kmeans`: elbow + silhouette sweep, cluster sizes, centroids, assignments.  
- `isoforest`: anomaly scores + top-N anomalies, score distribution.  
- `lof` (optional alternative).  
- `analyze`: per-cluster top-diff features; anomaly-feature correlations; mini `report.txt`.

---

## 📊 Example Results (from demo run in this repo)

- **Silhouette (embeddings)**:  
  - K=2 → **0.781**, K=3 → **0.779**, K=4 → 0.769 (2–3 clusters are natural)  
- **K=2 cluster sizes**: **126 vs 3** (tiny minority looks like a rare/atypical subgroup)  
- **IsolationForest anomalies**: ~10% flagged; **top scores ~0.75**.  
- **Interpretability** (*what you’ll see via `analyze`*):  
  - Cluster 1 (small) shows extreme lab signatures (e.g., high lactate/creatinine/glucose).  
  - Anomaly score correlates with the same features → **method agreement** (robust signal).

> Tip: Present **both** K=2 and K=3 (similar silhouettes). K=2 highlights a rare subgroup; K=3 can show a more nuanced gradient of phenotypes.

---

## 🧪 Validation & “Where are accuracy/F1?”

This is **unsupervised**. No labels → no accuracy/F1 by default. Evaluate via:

- **Internal**: reconstruction loss curves, silhouette, cluster sizes, anomaly score histograms.  
- **External (optional)**: join clusters/anomalies with downstream labels (e.g., `HOSPITAL_EXPIRE_FLAG`, LOS) to report **enrichment** or **precision/recall/F1** for a *specific outcome*.  
  - Example: *“Cluster 2 has 60% mortality vs 20% baseline”* → strong signal.

If you want a classic train/val/**test** split story:
- Split ICU stays (80/20); train VAE on train; infer on test; report test-only silhouette/anomaly patterns; optionally compute precision/recall/F1 if you bring in labels.

---

## 📈 Sensible Defaults & Scaling Tips

**Demo size (≈129×282):**
- VAE: `--hidden-sizes "256,128"`, `--latent-dim 8`, `--batch-size 32`, `--epochs 200`, `--patience 20`, `--beta 1.0`.
- KMeans: try K in 2–6; auto-select by silhouette, then inspect K=2 and K=3 explicitly.
- IF: `--n-estimators 300`, `--contamination 0.05–0.10`.

**Full MIMIC-III/IV:**
- VAE: upsize to `"512,256,128"`, latent 16–32; consider **AMP** (`torch.cuda.amp`) and **gradient clipping**.
- Clustering: `MiniBatchKMeans` for speed; consider **HDBSCAN** for non-spherical shapes.
- Anomalies: tune `max_samples`, contamination; try **IsolationForest** + **LOF/HBOS** + **Mahalanobis**.

---

## 🧷 Reproducibility

- Set seeds (`--seed`).
- Persist scalers/imputers/joblibs and `vae_config.json`.
- Log dropped non-numeric/all-NaN columns (console).
- Use pinned torch wheels (`+cpu` or `+cu124`) in `requirements.txt` or document separately.

---

## 📜 License & Acknowledgments

- License: MIT (or your preferred OSI license).  
- Acknowledgments: MIT Laboratory for Computational Physiology (MIMIC-III), PhysioNet; scikit-learn, PyTorch communities.

---


