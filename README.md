# Music Mood Identification  
**Fast, lightweight, fully-offline mood recogniser for music and long-form audio**

This project fuses a pure-PyTorch port of Google’s **YAMNet** backbone with a
custom 59-label MLP head that was trained on a large, human-curated mood
dataset.  It delivers:

* **Per-second mood probabilities** (0.96 s windows) for detailed timelines  
* **Whole-clip “theme”** via mean-pooled embeddings  
* Runs entirely on CPU or GPU — *no TensorFlow, no internet connectivity needed*  
* Reaches **state-of-the-art PR-AUC = 0.2934** on the held-out test split
  (surpassing published Jamendo/MTG mood benchmarks).
---

## Table of Contents
1. [Overview / Motivation](#overview--motivation)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Quick Start / Self-Test ▶️](#quick-start--self-test-)  
6. [Python API Usage](#python-api-usage)  
7. [Folder Structure](#folder-structure)  

---

## Overview / Motivation<a id="overview--motivation"></a>

Most music-tagging models give a single label for an entire track, yet songs can shift mood every few seconds.  
This project marries Google’s **YAMNet** backbone (pure **PyTorch** port) with a custom **59-tag MLP** to deliver:

* **Per-second predictions** (≈ 0 .96 s resolution)  
* **Whole-clip “theme”** via mean-pooled embeddings  
* **Fully offline inference** — no internet calls at runtime  

Great for playlist generation, DJ tools, radio automation, or large-scale catalog analysis.

---

## Features<a id="features"></a>

* **Pure-PyTorch YAMNet** backbone (MobileNet-style)  
* 59-tag **MLP** classifier (512 → 256 → 128 → 59 sigmoid)  
* **CLI** — one-line timeline or theme extraction  
* **Python API** — `predict_timeline_and_theme()` for easy integration  
* Runs on **CPU or GPU**; *no TensorFlow dependency*  
* Pre-trained weights already included in the repo:  
  * `yamnet_weights.pth` – YAMNet backbone weights  
  * `model_pytorch_model_v1_fold1.pt` – custom MLP head weights  

---

## Requirements<a id="requirements"></a>

| Package     | Tested version | Notes                                  |
|-------------|---------------|----------------------------------------|
| Python      | ≥ 3.10        |                                        |
| PyTorch     | 2.2.x         | CUDA 12.1 build recommended for GPU    |
| torchaudio  | 2.2.x         | Must match PyTorch version             |
| numpy       | ≥ 1.23        |                                        |
| soundfile   | ≥ 0.12        | Needed for non-WAV formats             |

---

## Installation<a id="installation"></a>

```bash
git clone https://github.com/your-org/music_mood_identification.git
cd music_mood_identification

# create & activate a virtual environment
python -m venv venv
source venv/bin/activate

# install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

```             
---

## Quick Start / Self-Test ▶️<a id="quick-start--self-test-"></a>

```bash
# Whole-clip mood theme (top-5 tags)
python yamnet_mlp_infer.py path/to/audio.wav --mood -k 5

# Timeline + theme (top-3 tags per ~1 s)
python yamnet_mlp_infer.py path/to/audio.wav --timeline -k 3

# Use GPU if available
python yamnet_mlp_infer.py path/to/audio.wav --timeline -k 3 --cuda

```

---

## Python API Usage<a id="python-api-usage"></a>

```python
from yamnet_mlp_infer import predict_timeline_and_theme

# Get per-second timeline and whole-clip theme
timeline, theme = predict_timeline_and_theme("path/to/audio.wav", top_k=3)

print("Whole-clip theme:")
for tag, prob in theme:
    print(f"{tag:25} {prob:.3f}")

print("\nFirst 5 seconds:")
for sec, sec_tags in timeline[:5]:
    tag_str = ", ".join(f"{t}:{p:.2f}" for t, p in sec_tags)
    print(f"{sec:6.2f}s  {tag_str}")

```
---

## Folder Structure<a id="folder-structure"></a>

```text
music_mood_identification/
├── yamnet_mlp_infer.py              # main inference script / API
├── yamnet_weights.pth               # YAMNet backbone weights
├── model_pytorch_model_v1_fold1.pt  # 59-tag MLP head weights
├── tag_index_mapping.txt            # index → tag mapping
├── requirements.txt                 # pip dependencies
├── code/
│   └── other python files            #(kept for provenance)
```