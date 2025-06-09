#!/usr/bin/env python3
"""
Drop rows whose track-IDs correspond to WAV files rejected as “>10 min”.
The error log has bare numbers (e.g., 948.wav); the TSV’s TRACK_ID
field stores IDs as 'track_0000948'. This script converts the numbers
accordingly and filters on column 0.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------- #
ERROR_LOG = Path("/mnt/data/Vineel/jamendo_project/code/yamnet_errors_v2.log")
TSV_IN    = Path("/mnt/data/Vineel/jamendo_project/autotagging_moodtheme-validation.tsv")
TSV_OUT   = TSV_IN.with_name(TSV_IN.stem + "_filtered.tsv")
ID_COL    = 0      # TRACK_ID column
PAD       = 7      # zero-pad width used in TRACK_ID
PREFIX    = "track_"
# ---------------------------------------------------------------------- #

# 1) Collect offending IDs in TSV format (track_000XXXX)
long_ids = set()
num_re = re.compile(r"/(\d+)\.wav")

with ERROR_LOG.open() as f:
    for line in f:
        m = num_re.search(line)
        if m:
            num = int(m.group(1))             # 948
            long_ids.add(f"{PREFIX}{num:0{PAD}d}")  # track_0000948

print(f"Will drop {len(long_ids):,} rows")

# 2) Stream-filter the TSV
kept, dropped = 0, 0
with TSV_IN.open() as fin, TSV_OUT.open("w") as fout:
    header = next(fin)            # copy header row
    fout.write(header)

    for ln in fin:
        if not ln.strip():
            continue
        track_id = ln.split("\t")[ID_COL]
        if track_id in long_ids:
            dropped += 1
        else:
            fout.write(ln)
            kept += 1

print(f"Finished → {TSV_OUT}")
print(f"  kept   : {kept:,}")
print(f"  dropped: {dropped:,}")
