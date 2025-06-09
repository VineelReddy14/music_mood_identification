#!/usr/bin/env python3
import os
import glob
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
AUDIO_DIR   = "/mnt/data/Vineel/jamendo_project/converted_wav"
OUTPUT_DIR  = "/mnt/data/Vineel/jamendo_project/yamnet_embeddings"
BATCH_SIZE  = 8   # on a 6-core/12-thread CPU + RTX3050, 8 is sweet spot
TARGET_SR   = 16000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────
print("Loading YAMNet on GPU…")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# ─── HELPERS ───────────────────────────────────────────────────────────
def load_wav(path):
    """Read WAV; enforce TARGET_SR."""
    wav, sr = sf.read(path, dtype="float32")
    if sr != TARGET_SR:
        raise ValueError(f"{path} has SR={sr}, expected {TARGET_SR}")
    return wav

# ─── COLLECT FILES ─────────────────────────────────────────────────────
wav_paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*", "*.wav")))
print(f"→ Found {len(wav_paths)} WAV files")

# ─── BATCHED EXTRACTION ────────────────────────────────────────────────
for i in tqdm(range(0, len(wav_paths), BATCH_SIZE), desc="Batches"):
    batch_paths = wav_paths[i : i + BATCH_SIZE]

    # 1) load all signals
    signals = []
    for p in batch_paths:
        try:
            signals.append(load_wav(p))
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
            signals.append(np.zeros((1,)))  # dummy so padding keeps indices

    # 2) pad to shape [batch, max_length]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        signals, padding="post", dtype="float32"
    )

    # 3) forward pass (outputs: scores, embeddings, spectrogram)
    _, embeddings, _ = yamnet(padded)  # embeddings: [B, T, 1024]

    # 4) mean-pool over time, cast to float16
    pooled = tf.reduce_mean(embeddings, axis=1).numpy().astype("float16")

    # 5) save each file
    for path, vec in zip(batch_paths, pooled):
        rel = os.path.relpath(path, AUDIO_DIR)
        out = os.path.join(OUTPUT_DIR, rel.replace(".wav", ".npy"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, vec)

print("\n✅ All embeddings saved to", OUTPUT_DIR)
