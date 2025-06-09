# phase1_make_pos_weight.py
# -------------------------
# Re-compute positive class weights with clipping
import os, numpy as np, pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cap", type=float, default=10.0,
                    help="maximum value allowed for any pos-weight element")
args = parser.parse_args()

base_dir   = "/mnt/data/Vineel/jamendo_project"
labels_csv = os.path.join(base_dir, "labels", "moodtheme_labels.csv")
out_path   = os.path.join(base_dir, "models", "pos_weight.npy")

df = pd.read_csv(labels_csv)
Y  = df.iloc[:, 2:].values              # multi-hot matrix   (N, 59)

pos_counts = Y.sum(axis=0)              # (#pos per tag,)
neg_counts = Y.shape[0] - pos_counts    # (#neg per tag,)

raw_pw   = neg_counts / (pos_counts + 1e-6)   # avoid /0
clipped  = np.clip(raw_pw, 1.0, args.cap)     # keep ≥1, cap high values
np.save(out_path, clipped.astype(np.float32))

print(f"✅  Saved pos_weight  (max={clipped.max():.1f})  →  {out_path}")