# Train_pytorch_model.py
# ----------------------
import os, argparse, logging, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
from Pytorch_model import MLP                     # same architecture, different dropouts
import matplotlib.pyplot as plt

# ───────── CLI ─────────
parser = argparse.ArgumentParser(description="5-fold CV with clipped pos_weight")
parser.add_argument("--run_name",   required=True, type=str)
parser.add_argument("--epochs",     default=60, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--patience",   default=8,  type=int)
parser.add_argument("--folds",      default=5,  type=int)
# dropout defaults now a bit stronger
parser.add_argument("--dropout1",   default=0.4, type=float)
parser.add_argument("--dropout2",   default=0.4, type=float)
parser.add_argument("--dropout3",   default=0.3, type=float)
args = parser.parse_args()

# ───────── paths & logging ─────────
base = "/mnt/data/Vineel/jamendo_project"
logs = os.path.join(base, "log");     os.makedirs(logs,   exist_ok=True)
plots= os.path.join(base, "plots");   os.makedirs(plots,  exist_ok=True)
models= os.path.join(base,"models");  os.makedirs(models, exist_ok=True)

log_file = os.path.join(logs, f"log_{args.run_name}.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")
print(f"✓ logging to {log_file}")

# ───────── load data ─────────
csv_path  = os.path.join(base, "labels", "moodtheme_labels.csv")
emb_dir   = os.path.join(base, "yamnet_embeddings_v3")
pos_path  = os.path.join(base, "models", "pos_weight.npy")

df = pd.read_csv(csv_path)
X_raw  = np.array([np.load(os.path.join(emb_dir, p)) for p in df["path"]])
y_raw  = df.iloc[:, 2:].values.astype(np.float32)
y_str  = y_raw.argmax(1)                    # for stratification

# --- standardise embeddings (μ & σ from whole set) ---
mu  = X_raw.mean(axis=0, keepdims=True)
std = X_raw.std(axis=0,  keepdims=True) + 1e-6
X_raw = ((X_raw - mu) / std).astype(np.float32)

# pos_weight vector
pos_weight = torch.tensor(np.load(pos_path), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ using device: {device}")

# ───────── containers for plots ─────────
all_metrics = {m: [] for m in ["loss", "pr_auc"]}

# ───────── k-fold loop ─────────
cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(cv.split(X_raw, y_str), 1):
    print(f"\n🔁  Fold {fold}/{args.folds}")
    Xtr = torch.tensor(X_raw[tr]);   ytr = torch.tensor(y_raw[tr])
    Xva = torch.tensor(X_raw[va]);   yva = torch.tensor(y_raw[va])

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size)

    model = MLP(1024, y_raw.shape[1], args.dropout1, args.dropout2, args.dropout3).to(device)
    opt   = optim.Adam(model.parameters(), lr=3e-4)
    cri   = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_auc, wait, hist_loss, hist_auc = 0.0, 0, [], []

    for ep in range(1, args.epochs+1):
        # --- train ---
        model.train(); ep_loss = 0.
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)                       # logits
            loss = cri(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(train_dl)

        # --- val ---
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb.to(device))
                all_logits.append(torch.sigmoid(logits).cpu())
                all_labels.append(yb)
        preds  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        pr_auc = average_precision_score(labels, preds, average="macro")
        hist_loss.append(ep_loss);  hist_auc.append(pr_auc)

        print(f"Ep {ep:02d}  tr_loss={ep_loss:.4f}  va_pr_auc={pr_auc:.3f}")
        logging.info(f"Fold {fold} Ep {ep} loss={ep_loss:.4f} pr_auc={pr_auc:.3f}")

        # --- early stop on PR-AUC ---
        if pr_auc > best_auc:
            best_auc, wait = pr_auc, 0
            torch.save(model.state_dict(),
                       os.path.join(models, f"model_{args.run_name}_fold{fold}.pt"))
        else:
            wait += 1
            if wait >= args.patience:
                break
        # lr scheduler (simple)
        if (ep % 4 == 0) and (wait > 0):
            for g in opt.param_groups:
                g["lr"] *= 0.5

    # collect curve for combined plot
    all_metrics["loss"].append(hist_loss)
    all_metrics["pr_auc"].append(hist_auc)

# ───────── combined plots ─────────
for metric, curves in all_metrics.items():
    plt.figure(figsize=(6,4))
    for i, c in enumerate(curves, 1):
        plt.plot(c, label=f"fold{i}")
    plt.xlabel("epoch"); plt.ylabel(metric)
    plt.title(f"{metric.upper()} across folds")
    plt.legend()
    out_png = os.path.join(plots, f"{args.run_name}_{metric}.png")
    plt.savefig(out_png); plt.close()
    print(f"📊  saved {out_png}")

print("✓ training finished")