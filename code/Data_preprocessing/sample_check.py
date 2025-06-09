import os
import soundfile as sf

AUDIO_DIR = "/mnt/data/Vineel/jamendo_project/converted_wav"
bad_sr = []

for root, _, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            try:
                _, sr = sf.read(path)
                if sr != 16000:
                    bad_sr.append((path, sr))
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

if not bad_sr:
    print("✅ All files are 16kHz.")
else:
    print(f"❌ Found {len(bad_sr)} files not 16kHz.")
    for path, sr in bad_sr[:5]:
        print(f"{path}: {sr} Hz")
