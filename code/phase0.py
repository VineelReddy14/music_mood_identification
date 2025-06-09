# phase0.py
"""
Phase-0 audit:  per-tag positive counts & percentage.
Outputs:
  • tag_frequency.csv         (Tag, Positives, Frequency)
  • tag_frequency.png         horizontal bar-plot (optional)
"""

import os, pandas as pd, matplotlib.pyplot as plt

BASE = "/mnt/data/Vineel/jamendo_project"
LABEL_CSV   = f"{BASE}/labels/moodtheme_labels.csv"
MAPPING_TXT = f"{BASE}/tag_index_mapping.txt"      # optional
OUT_DIR     = f"{BASE}/log"
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── 1. load label matrix ─────────
df = pd.read_csv(LABEL_CSV)          # columns: track_id | path | tag0 … tag58
Y = df.iloc[:, 2:].copy()            # only multi-hot part
N_tracks = len(Y)

# ───────── 2. optional: load tag names ─────────
try:
    tag_names = [l.split('\t',1)[-1].strip()
                 for l in open(MAPPING_TXT, encoding="utf-8")]
    if len(tag_names) != Y.shape[1]:
        raise ValueError
except Exception:
    tag_names = [f"Tag {i}" for i in range(Y.shape[1])]

Y.columns = tag_names                # nice column headers

# ───────── 3. compute frequency ─────────
pos_counts = Y.sum()                 # series length 59
freq = pos_counts / N_tracks

freq_df = (pd.DataFrame({
            "Tag"        : tag_names,
            "Positives"  : pos_counts.values.astype(int),
            "Frequency"  : freq.values })
          .sort_values("Positives", ascending=False))

csv_path = os.path.join(OUT_DIR, "tag_frequency.csv")
freq_df.to_csv(csv_path, index=False)
print(f"✅  Saved {csv_path}")

# ───────── 4. (optional) bar-plot ─────────
plt.figure(figsize=(8, 14))
plt.barh(freq_df["Tag"], freq_df["Positives"])
plt.gca().invert_yaxis()
plt.title("Jamendo mood/theme – positive counts per tag")
plt.xlabel("# positive tracks")
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "tag_frequency.png")
plt.savefig(png_path, dpi=150)
print(f"✅  Saved {png_path}")
