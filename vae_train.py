# save as vae_train.py
import argparse
import json
import os
import sys
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

# Optional deps (graceful fallbacks)
def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[WARN] matplotlib not available ({e}). Skipping plots. To enable: pip install matplotlib")
        return None

def try_import_umap():
    try:
        import umap
        return umap
    except Exception as e:
        print(f"[INFO] umap-learn not available ({e}). Will fall back to PCA for 2D projection if possible. To enable: pip install umap-learn")
        return None

# Light sklearn utilities (silently assumed available per your env)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Torch (hard requirement for this script)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    print(f"[ERROR] PyTorch is not installed ({e}). Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)


# -------------------- utils --------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_hidden(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]

def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only columns (drop all-NaN)."""
    num = df.select_dtypes(include=[np.number]).copy()
    return num.loc[:, num.notna().any(axis=0)]

def split_and_scale(
    X: np.ndarray,
    test_size: float,
    seed: int,
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    X_train, X_val = train_test_split(X, test_size=test_size, random_state=seed, shuffle=True)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    return X_train, X_val, X_train.copy(), X_val.copy(), scaler  # last two are originals if needed later

def to_tensors(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32, device=device)

def kld_loss(mu, logvar):
    # KL divergence between N(mu, sigma) and N(0,1)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # per-sample

def mse_recon(x, x_recon):
    return torch.mean((x - x_recon) ** 2, dim=1)  # per-sample

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------- VAE model --------------------

class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, latent_dim) if hidden else nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim) if hidden else nn.Linear(input_dim, latent_dim)

        # Decoder
        dec_layers = []
        prev_d = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev_d, h), nn.ReLU()]
            prev_d = h
        self.decoder = nn.Sequential(*dec_layers)
        self.out = nn.Linear(prev_d, input_dim)

    def encode(self, x):
        if len(self.encoder) == 0:
            h = x
        else:
            h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        if len(self.decoder) == 0:
            h = z
        else:
            h = self.decoder(z)
        return self.out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

# -------------------- training / inference --------------------

def train_vae(
    X: np.ndarray,
    ids: pd.Series,
    outdir: str,
    *,
    id_col: str,
    hidden: List[int],
    latent_dim: int,
    beta: float,
    lr: float,
    batch_size: int,
    weight_decay: float,
    epochs: int,
    val_split: float,
    patience: int,
    seed: int,
    standardize: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    ensure_dir(outdir)

    # split + scale
    X_train, X_val, _, _, scaler = split_and_scale(
        X, test_size=val_split, seed=seed, standardize=standardize
    )

    # data loaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # model/optim
    model = MLPVAE(input_dim=X.shape[1], hidden=hidden, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = math.inf
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            x_recon, mu, logvar, _ = model(xb)
            recon = mse_recon(xb, x_recon)  # per-sample
            kld = kld_loss(mu, logvar)
            loss = torch.mean(recon + beta * kld)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_recon, mu, logvar, _ = model(xb)
                recon = mse_recon(xb, x_recon)
                kld = kld_loss(mu, logvar)
                loss = torch.mean(recon + beta * kld)
                val_losses.append(loss.item())

        tr_mean = float(np.mean(tr_losses)) if tr_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(tr_mean)
        history["val_loss"].append(val_mean)
        print(f"[EP {ep:03d}] train={tr_mean:.6f}  val={val_mean:.6f}")

        if val_mean < best_val - 1e-6:
            best_val = val_mean
            wait = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"[EARLY STOP] No val improvement for {patience} epochs. Best val={best_val:.6f}")
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(outdir, "vae_model.pt"))
    cfg = {
        "id_col": id_col,
        "input_dim": int(X.shape[1]),
        "hidden": hidden,
        "latent_dim": int(latent_dim),
        "beta": float(beta),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "val_split": float(val_split),
        "patience": int(patience),
        "seed": int(seed),
        "standardize": bool(standardize),
        "device": str(device),
    }
    save_json(cfg, os.path.join(outdir, "vae_config.json"))

    if scaler is not None:
        dump(scaler, os.path.join(outdir, "scaler.joblib"))
    # Save feature column order too
    save_json({"feature_columns": list(range(X.shape[1]))}, os.path.join(outdir, "feature_columns.json"))

    # Full-pass to export embeddings & recon error (use all data, with scaler if any)
    if standardize and scaler is not None:
        X_all = scaler.transform(X)
    else:
        X_all = X.copy()

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_all, dtype=torch.float32, device=device)
        x_recon, mu, logvar, z = model(X_tensor)
        recon_err = torch.mean((X_tensor - x_recon) ** 2, dim=1).cpu().numpy()
        emb = mu.cpu().numpy()  # use mu as embedding (stable)

    # Save CSVs
    emb_df = pd.DataFrame(emb, columns=[f"z{i+1}" for i in range(emb.shape[1])])
    emb_df.insert(0, cfg["id_col"], ids.values)
    emb_df.to_csv(os.path.join(outdir, "embeddings.csv"), index=False)

    re_df = pd.DataFrame({
        cfg["id_col"]: ids.values,
        "reconstruction_mse": recon_err
    })
    re_df.to_csv(os.path.join(outdir, "recon_error.csv"), index=False)

    # Plots
    plt = try_import_matplotlib()
    if plt is not None:
        # loss curves
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(history["train_loss"], label="train")
        ax.plot(history["val_loss"], label="val")
        ax.set_title("β-VAE loss (MSE + β·KL)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "loss_curves.png"), dpi=150)
        plt.close(fig)

        # recon error histogram
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(recon_err, bins=30)
        ax.set_title("Reconstruction error (MSE)")
        ax.set_xlabel("MSE")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "recon_error_hist.png"), dpi=150)
        plt.close(fig)

        # 2D projection of embeddings
        proj = None
        umap = try_import_umap()
        if umap is not None and emb.shape[1] >= 2:
            reducer = umap.UMAP(n_components=2, random_state=seed)
            proj = reducer.fit_transform(emb)
        elif emb.shape[1] >= 2:
            p = PCA(n_components=2, random_state=seed)
            proj = p.fit_transform(emb)

        if proj is not None:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(proj[:,0], proj[:,1], s=12)
            ax.set_title("Embeddings (2D projection)")
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "embeddings_2d.png"), dpi=150)
            plt.close(fig)

    print(f"[DONE] Artifacts saved to: {outdir}")


def run_infer(
    features_path: str,
    outdir: str,
    id_col: str,
    model_path: str,
    standardize: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    df = pd.read_csv(features_path)
    ids = df[id_col] if id_col in df.columns else pd.Series(np.arange(len(df)), name=id_col)
    X = safe_numeric_df(df.drop(columns=[id_col], errors="ignore")).values.astype(np.float32)

    # load config & model
    cfg_path = os.path.join(os.path.dirname(model_path), "vae_config.json")
    if not os.path.exists(cfg_path):
        print(f"[WARN] Could not find config at {cfg_path}; proceeding with CLI flags.")
        input_dim_guess = X.shape[1]
        hidden = []
        latent_dim = 8
    else:
        cfg = json.load(open(cfg_path, "r"))
        input_dim_guess = cfg.get("input_dim", X.shape[1])
        hidden = cfg.get("hidden", [])
        latent_dim = cfg.get("latent_dim", 8)
        standardize = cfg.get("standardize", standardize)

    model = MLPVAE(input_dim=input_dim_guess, hidden=hidden, latent_dim=latent_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scaler = None
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler.joblib")
    if standardize and os.path.exists(scaler_path):
        scaler = load(scaler_path)

    X_proc = scaler.transform(X) if (standardize and scaler is not None) else X.copy()

    with torch.no_grad():
        X_tensor = torch.tensor(X_proc, dtype=torch.float32, device=device)
        x_recon, mu, logvar, z = model(X_tensor)
        recon_err = torch.mean((X_tensor - x_recon) ** 2, dim=1).cpu().numpy()
        emb = mu.cpu().numpy()

    emb_df = pd.DataFrame(emb, columns=[f"z{i+1}" for i in range(emb.shape[1])])
    emb_df.insert(0, id_col, ids.values)
    emb_df.to_csv(os.path.join(outdir, "embeddings.csv"), index=False)

    re_df = pd.DataFrame({id_col: ids.values, "reconstruction_mse": recon_err})
    re_df.to_csv(os.path.join(outdir, "recon_error.csv"), index=False)

    print(f"[INFER DONE] embeddings.csv and recon_error.csv saved to: {outdir}")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="β-VAE for ICU phenotype discovery (MIMIC-III, first 24h)")
    parser.add_argument("--features", required=True, help="Path to features_clean.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for artifacts (a 'vae' subfolder will be created)")
    parser.add_argument("--id-col", default="icustay_id", help="Identifier column name (default: icustay_id)")

    # model + train
    parser.add_argument("--hidden-sizes", default="256,128", help="Comma-separated hidden sizes, e.g. '256,128'")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", type=lambda s: s.lower() in {"1","true","yes"}, default=True)

    # infer mode
    parser.add_argument("--infer", action="store_true", help="Run in inference-only mode (requires --model-path)")
    parser.add_argument("--model-path", default="", help="Path to vae_model.pt for --infer")

    args = parser.parse_args()

    outdir = ensure_dir(os.path.join(args.outdir, "vae"))

    if args.infer:
        if not args.model_path:
            print("[ERROR] --infer requires --model-path pointing to vae_model.pt")
            sys.exit(2)
        run_infer(
            features_path=args.features,
            outdir=outdir,
            id_col=args.id_col,
            model_path=args.model_path,
            standardize=args.standardize,
        )
        return

    # training path
    df = pd.read_csv(args.features)
    if args.id_col in df.columns:
        ids = df[args.id_col]
    else:
        print(f"[WARN] id column '{args.id_col}' not found; using 0..N-1.")
        ids = pd.Series(np.arange(len(df)), name=args.id_col)

    # numeric data
    X_df = safe_numeric_df(df.drop(columns=[args.id_col], errors="ignore"))
    dropped = [c for c in df.columns if c not in X_df.columns and c != args.id_col]
    if dropped:
        print(f"[INFO] Dropped non-numeric or all-NaN columns ({len(dropped)}): {dropped[:8]}{' ...' if len(dropped)>8 else ''}")

    X = X_df.values.astype(np.float32)
    if X.shape[0] < 2 or X.shape[1] < 1:
        print("[ERROR] Not enough data to train VAE.")
        sys.exit(3)

    hidden = parse_hidden(args.hidden_sizes)

    train_vae(
        X=X,
        ids=ids,
        outdir=outdir,
        id_col=args.id_col,
        hidden=hidden,
        latent_dim=args.latent_dim,
        beta=args.beta,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        val_split=args.val_split,
        patience=args.patience,
        seed=args.seed,
        standardize=bool(args.standardize),
    )

if __name__ == "__main__":
    main()
