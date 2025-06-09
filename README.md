
---

# Jamendo Mood-Classification Inference Guide

A minimal, end-to-end recipe for turning an audio file into **59 mood/theme probabilities** using:

1. a **PyTorch port of YAMNet** for embedding extraction
2. the provided **MLP** for classification

---

## Step 1 – Load the audio you want to classify

### 1-a) Pick (or record) an audio file

* Any common format is fine (`.wav`, `.mp3`, `.flac`, …).
* Longer clips also work; we’ll slice them into YAMNet-sized patches later.

### 1-b) Read the waveform into Python

```python
import torchaudio

wav_path = "path/to/your_audio.wav"      # ← replace
waveform, sample_rate = torchaudio.load(wav_path)  # Tensor [channels, samples]
print(waveform.shape, sample_rate)       # e.g. torch.Size([2, 480000]) 48000
```

### 1-c) (Optional) Quick sanity-check

```python
import IPython.display as ipd
ipd.Audio(waveform.numpy(), rate=sample_rate)
```

You now have a `waveform` tensor and its `sample_rate` ready for **Step 2**.

---

## Step 2 – Extract frame-level embeddings with YAMNet

> **YAMNet** expects **mono 16-kHz** audio and returns a **1 024-D embedding** every **0.96 s** (50 % overlap).

### 2-a) Resample & convert to mono

```python
import torch, torchaudio

# ── ensure mono ──
if waveform.shape[0] > 1:          # stereo → mono
    waveform = waveform.mean(dim=0, keepdim=True)

# ── ensure 16 kHz ──
if sample_rate != 16_000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                               new_freq=16_000)
    waveform = resampler(waveform)
    sample_rate = 16_000
```

### 2-b) Turn the waveform into YAMNet “patches”

```python
from torch_vggish_yamnet.input_proc import waveform_to_examples
import torch

examples = torch.from_numpy(
    waveform_to_examples(
        waveform.squeeze().numpy().astype("float32"),  # 1-D float32
        sample_rate                                    # 16 000
    )
)   # → [N, 1, 64, 96]
print("patch tensor shape:", examples.shape)
```

Each patch (96 frames × 64 mel bins) covers **0.96 s**; consecutive patches overlap by **50 %**.

### 2-c) Run YAMNet on every patch

```python
from torch_vggish_yamnet.yamnet.model import yamnet as YAMNet
import torch.nn.functional as F
import torch

yamnet = YAMNet()
yamnet.load_state_dict(
    torch.load("/mnt/data/Vineel/jamendo_project/models/yamnet_pytorch_weights.pth",
               map_location="cpu")         # ← adjust if needed
)
yamnet.eval()

with torch.no_grad():
    embeddings, *_ = yamnet(examples)      # [N, 1024, 1, 1]
    embeddings = embeddings.squeeze(-1).squeeze(-1)  # [N, 1024]
print("embedding matrix:", embeddings.shape)
```

You now have `embeddings`—one 1 024-D vector per **0.96 s** patch.

---

## Step 3 – Mean-pool the patch embeddings

Our MLP expects **one** 1 024-D vector per track.
The simplest approach (and what we used in training) is to average the patch embeddings:

```python
# embeddings : Tensor [N_patches, 1024]
track_embedding = embeddings.mean(dim=0, keepdim=True)   # shape → [1, 1024]
print("mean-pooled embedding:", track_embedding.shape)
```

---

## Step 4 – Run the MLP & decode the tag names

Goal: turn the **1 × 1024** `track_embedding` into **59 mood/theme probabilities** and show the top-k tags.

```python
import torch, numpy as np
from Pytorch_model import MLP   # same class used in training

# ── paths ──────────────────────────────────────────
TAG_FILE = "tag_index_mapping.txt"               # shipped with the bundle
MLP_PT   = "model_pytorch_model_v1_fold1.pt"      # MLP weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) load tag names
def load_tag_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t", 1)[-1] for line in f]

tag_names = load_tag_names(TAG_FILE)
assert len(tag_names) == 59

# 2) load the MLP
mlp = MLP(input_dim=1024, output_dim=len(tag_names)).to(DEVICE)
mlp.load_state_dict(torch.load(MLP_PT, map_location=DEVICE))
mlp.eval()

# 3) probabilities
with torch.no_grad():
    probs = mlp(track_embedding.to(DEVICE)).squeeze(0).cpu().numpy()

# 4) show top-k
TOP_K = 5
top_idx = probs.argsort()[-TOP_K:][::-1]

print(f"\nTop-{TOP_K} predicted moods/themes:")
for i in top_idx:
    print(f"  • {tag_names[i]:<30} {probs[i]:.2f}")
```

---

## File List in This Bundle

| File                                    | Purpose                                   |
| --------------------------------------- | ----------------------------------------- |
| **yamnet\_pytorch\_weights.pth**        | YAMNet weights (PyTorch)                  |
| **model\_pytorch\_model\_v1\_fold1.pt** | Trained MLP weights                       |
| **tag\_index\_mapping.txt**             | Maps the 59 indices to mood/theme strings |

---

### Citation

If you use this code or model in academic work, please cite:

* **YAMNet** (original TensorFlow version)
* **MTG-Jamendo** dataset
* Your own project/report

---

*Last updated: 2025-06-08*
