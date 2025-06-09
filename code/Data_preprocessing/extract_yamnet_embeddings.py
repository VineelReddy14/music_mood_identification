import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

AUDIO_DIR = "/mnt/data/Vineel/jamendo_project/converted_wav"
OUTPUT_DIR = "/mnt/data/Vineel/jamendo_project/yamnet_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YAMNet model once in parent process (for GPU)...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Get list of all .wav files
wav_files = []
for subdir in sorted(os.listdir(AUDIO_DIR)):
    subdir_path = os.path.join(AUDIO_DIR, subdir)
    if os.path.isdir(subdir_path):
        for fname in os.listdir(subdir_path):
            if fname.endswith(".wav"):
                wav_files.append(os.path.join(subdir_path, fname))


def process_file(file_path):
    rel_path = os.path.relpath(file_path, AUDIO_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel_path.replace(".wav", ".npy"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        return  # already processed

    try:
        wav_data, sr = sf.read(file_path)
        if sr != 16000:
            raise ValueError(f"Expected 16kHz, got {sr}")

        # Run YAMNet
        _, embeddings, _ = yamnet_model(wav_data)
        np.save(out_path, embeddings.numpy())

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")


if __name__ == "__main__":
    print(f"🔍 Found {len(wav_files)} WAV files.")
    print(f"🚀 Using {cpu_count()} CPU cores for parallel processing.\n")

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, wav_files), total=len(wav_files),
                  desc="Extracting YAMNet embeddings"))

    print(f"\n✅ All embeddings saved to: {OUTPUT_DIR}")
