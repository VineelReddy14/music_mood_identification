import os
# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import openl3
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pathlib

# Set your input/output directories
input_base = "/mnt/data/Vineel/jamendo_project/converted_wav"
output_base = "/mnt/data/Vineel/jamendo_project/openl3_embeddings"
os.makedirs(output_base, exist_ok=True)

# Model settings
content_type = "music"
input_repr = "mel256"
embedding_size = 6144  # or 512
hop_size = 0.1  # in seconds

# Collect folders (e.g., 00 to 99)
folders = sorted([f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))])
num_folders = len(folders)
embedding_counts = []

def extract_openl3_embedding(filepath, output_path):
    try:
        audio, sr = sf.read(filepath)
        emb, ts = openl3.get_audio_embedding(audio, sr,
                                             input_repr=input_repr,
                                             content_type=content_type,
                                             embedding_size=embedding_size,
                                             hop_size=hop_size)
        np.savez(output_path, embedding=emb, timestamps=ts)
        return True
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

print(f"Total folders: {num_folders}")
start_time = time.time()

# Loop over all folders and extract embeddings
for folder in tqdm(folders, desc="Folders"):
    folder_path = os.path.join(input_base, folder)
    output_folder = os.path.join(output_base, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    files = sorted(pathlib.Path(folder_path).glob("*.wav"))
    success_count = 0

    for file_path in tqdm(files, desc=folder, leave=False):
        output_file = os.path.join(output_folder, file_path.stem + ".npz")
        if not os.path.exists(output_file):
            success = extract_openl3_embedding(str(file_path), output_file)
            if success:
                success_count += 1
    
    embedding_counts.append(success_count)

end_time = time.time()
print(f"Completed in {end_time - start_time:.2f} seconds")

# Plot progress
plt.figure(figsize=(10, 5))
plt.bar(range(num_folders), embedding_counts)
plt.xlabel("Folder Index")
plt.ylabel("Successful Embeddings")
plt.title("OpenL3 Embeddings Extracted per Folder")
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/Vineel/jamendo_project/embedding_progress_chart.png")
plt.show()