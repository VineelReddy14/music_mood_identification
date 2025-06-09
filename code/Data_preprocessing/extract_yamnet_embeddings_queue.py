#!/usr/bin/env python3
"""
Multiprocess YAMNet extractor — safe on a 4 GB RTX 3050.
CPU workers run purely on CPU,  GPU worker processes ONE file at a time.
"""

import os, sys, time, math, pathlib, multiprocessing as mp
from queue import Empty

# ─────────── paths ────────────────────────────────────────────────────────────
AUDIO_DIR = pathlib.Path("/mnt/data/Vineel/jamendo_project/converted_wav")
OUT_DIR   = pathlib.Path("/mnt/data/Vineel/jamendo_project/yamnet_embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────── knobs ────────────────────────────────────────────────────────────
CPU_WORKERS   = 6      # good for a 6-core / 12-thread Ryzen 5
QUEUE_SIZE    = 512
CLEAR_EVERY_N = 250    # clear GPU TF graph every N files

# ------------------------------------------------------------------------------
def list_wavs():
    return sorted(AUDIO_DIR.rglob("*.wav"))

def out_path(wav):
    return OUT_DIR / wav.relative_to(AUDIO_DIR).with_suffix(".npy")

def save_np(arr, path):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype("float16"))

# ------------------------------------------------------------------------------
def load_yamnet(device="/CPU:0"):
    import tensorflow as tf, tensorflow_hub as hub
    with tf.device(device):
        return hub.load("https://tfhub.dev/google/yamnet/1")

# CPU -------------------------------------------------------------------------
def cpu_worker(q, done):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""       # hide GPUs
    import soundfile as sf
    model = load_yamnet("/CPU:0")

    while True:
        try:
            wav = q.get(timeout=5)
        except Empty: break
        if wav is None: break

        try:
            out = out_path(wav)
            if out.exists():
                done.put(1);  continue
            data, sr = sf.read(wav)
            if sr != 16000:
                raise ValueError(f"{sr} Hz (needs 16 kHz)")
            _, emb, _ = model(data)
            save_np(emb.numpy().mean(axis=0), out)
        except Exception as e:
            sys.stderr.write(f"[CPU] {wav}: {e}\n")
        done.put(1)

# GPU -------------------------------------------------------------------------
def gpu_worker(q, done):
    import tensorflow as tf, soundfile as sf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        sys.stderr.write("[GPU] No CUDA device visible!\n")
        return
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model = load_yamnet("/GPU:0")

    handled = 0
    while True:
        try:
            wav = q.get(timeout=5)
        except Empty: break
        if wav is None: break

        try:
            out = out_path(wav)
            if out.exists():
                done.put(1);  continue
            data, sr = sf.read(wav)
            if sr != 16000:
                raise ValueError(f"{sr} Hz (needs 16 kHz)")
            with tf.device("/GPU:0"):
                _, emb, _ = model(data)           # 1-D waveform
            save_np(emb.numpy().mean(axis=0), out)
            handled += 1
            if handled % CLEAR_EVERY_N == 0:
                tf.keras.backend.clear_session()  # free cuFFT scratch
        except Exception as e:
            sys.stderr.write(f"[GPU] {wav}: {e}\n")
        done.put(1)

# ─────────── main ─────────────────────────────────────────────────────────────
def main():
    pending = [w for w in list_wavs() if not out_path(w).exists()]
    total   = len(pending)
    if not total:
        print("✓ Nothing to do — all embeddings exist.");  return
    print(f"→ {total:,} WAV files to process")

    mp.set_start_method("spawn", force=True)
    work_q = mp.Queue(maxsize=QUEUE_SIZE)
    done_q = mp.Queue()

    # workers
    cpu_ps = [mp.Process(target=cpu_worker, args=(work_q, done_q))
              for _ in range(CPU_WORKERS)]
    gpu_p  = mp.Process(target=gpu_worker, args=(work_q, done_q))

    for p in cpu_ps + [gpu_p]:
        p.daemon = True;  p.start()

    # enqueue work
    for w in pending: work_q.put(w)
    for _ in cpu_ps + [gpu_p]: work_q.put(None)  # poison pills

    # progress
    done = 0;  t0 = time.time()
    while done < total:
        done += done_q.get()
        if done % 100 == 0 or done == total:
            rate = done / max(1e-6, time.time() - t0)
            eta  = (total - done) / rate / 60
            pct  = 100 * done / total
            print(f"\r{done}/{total}  {pct:5.1f}%  {rate:5.2f} f/s  ETA {eta:5.1f} min", end="")
    print("\n✓ Finished.")

    for p in cpu_ps + [gpu_p]: p.join()

if __name__ == "__main__":
    main()
