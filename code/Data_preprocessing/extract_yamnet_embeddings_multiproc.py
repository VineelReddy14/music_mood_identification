#!/usr/bin/env python3
"""
extract_yamnet_embeddings_mixed_v2.py
– 1 GPU worker + N_CPU workers
Writes mean-pooled 1024-D YAMNet embeddings atomically.
"""

import os, multiprocessing as mp, numpy as np, soundfile as sf
import tensorflow as tf, tensorflow_hub as hub
from tqdm import tqdm

# --------------------------------------------------------------------------- #
AUDIO_DIR   = "/mnt/data/Vineel/jamendo_project/converted_wav"
OUT_DIR_NEW = "/mnt/data/Vineel/jamendo_project/yamnet_embeddings_v3"
NUM_CPU     = 4
TFHUB_CACHE = "/mnt/data/tfhub_cache"
MAX_SECS    = 600        # skip >10-minute clips

os.makedirs(OUT_DIR_NEW, exist_ok=True)
os.makedirs(TFHUB_CACHE, exist_ok=True)
# --------------------------------------------------------------------------- #

def list_wavs(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".wav"):
                yield os.path.join(dirpath, f)

def wav_to_npy_path(wav):
    rel = os.path.relpath(wav, AUDIO_DIR)
    return os.path.join(OUT_DIR_NEW, rel[:-4] + ".npy")

# --------------------------- worker init ----------------------------------- #
def _load_model():
    os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE
    return hub.load("https://tfhub.dev/google/yamnet/1")

def init_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    global yamnet
    yamnet = _load_model()
    print(f"[GPU worker {os.getpid()}] model loaded")

def init_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global yamnet
    yamnet = _load_model()
    print(f"[CPU worker {os.getpid()}] model loaded (CPU)")

# --------------------------- per-file job ---------------------------------- #
def process_file(wav_path):
    out_path = wav_to_npy_path(wav_path)
    if os.path.exists(out_path):
        return "skip"                               # already done

    try:
        wav, sr = sf.read(wav_path, dtype='float32')
        if sr != 16_000:
            raise RuntimeError(f"Sample-rate {sr} ≠ 16 kHz")
        if wav.shape[0] > sr * MAX_SECS:
            raise RuntimeError("Clip longer than 10 min")

        _, emb, _ = yamnet(wav)
        vec = emb.numpy().mean(axis=0).astype(np.float16)

        # ensure sub-folder exists  ← NEW LINE
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, vec)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, out_path)                   # atomic rename
        return "ok"

    except Exception as e:
        return f"{wav_path}: {e}"

# ------------------------------ main --------------------------------------- #
if __name__ == "__main__":
    all_wavs  = list(list_wavs(AUDIO_DIR))
    pending   = [w for w in all_wavs if not os.path.exists(wav_to_npy_path(w))]
    print(f"→ {len(pending):,} WAVs need embeddings")

    gpu_pool = mp.Pool(1,       initializer=init_gpu)
    cpu_pool = mp.Pool(NUM_CPU, initializer=init_cpu)

    gpu_jobs = pending[0::NUM_CPU + 1]
    cpu_jobs = [w for w in pending if w not in gpu_jobs]

    stats, errors = {"ok": 0, "skip": 0}, []

    for res in tqdm(gpu_pool.imap_unordered(process_file, gpu_jobs),
                    total=len(gpu_jobs), desc="GPU worker"):
        if res in stats: stats[res] += 1
        else:            errors.append(res)

    for res in tqdm(cpu_pool.imap_unordered(process_file, cpu_jobs),
                    total=len(cpu_jobs), desc="CPU workers"):
        if res in stats: stats[res] += 1
        else:            errors.append(res)

    gpu_pool.close(); cpu_pool.close()
    gpu_pool.join();  cpu_pool.join()

    msg = (f"\nFinished YAMNet embedding extraction → {OUT_DIR_NEW}\n"
           f"  processed: {stats['ok']:,}\n"
           f"  skipped:   {stats['skip']:,}\n"
           f"  errors:    {len(errors):,}")
    print(msg)

    if errors:
        err_file = "yamnet_errors_v2.log"
        with open(err_file, "w") as f:
            f.write("\n".join(errors))
        print(f"  • see {err_file} for details")
