# tester.py  –  YAMNet (PyTorch) ➜ mean‑pool ➜ custom MLP
# Uses the **previous** MLP checkpoint (model_pytorch_model_v1_fold1.pt)
# Adds numerical‑safety guards to avoid NaNs/Inf and keeps all probabilities in [0,1].

import os, torch, torchaudio, numpy as np
from torch_vggish_yamnet.yamnet.model import yamnet as YAMNet
from Pytorch_model import MLP

# ───────── user paths ─────────
BASE        = "/mnt/data/Vineel/jamendo_project"
yamnet_wts  = f"{BASE}/models/yamnet_pytorch_weights.pth"
mlp_wts     = f"{BASE}/models/model_pytorch_model_v1_fold1.pt"   # ← back‑to‑old model
audio_dir   = f"{BASE}/epidemics audio files"
mapping_txt = f"{BASE}/tag_index_mapping.txt"

# ───────── tag names ─────────

def load_tag_names(path:str):
    names = []
    with open(path, "r", encoding="utf‑8") as f:
        for line in f:
            names.append(line.strip().split('\t',1)[-1])
    if len(names)!=59:
        raise ValueError(f"Expected 59 tags, got {len(names)} in {path}")
    return names

label_names = load_tag_names(mapping_txt)
print(f"  loaded {len(label_names)} mood/theme tags")

# ───────── load models ─────────
yamnet = YAMNet(); yamnet.load_state_dict(torch.load(yamnet_wts, map_location="cpu")); yamnet.eval()
mlp    = MLP(input_dim=1024, output_dim=len(label_names))
mlp.load_state_dict(torch.load(mlp_wts, map_location="cpu")); mlp.eval()
print("  models loaded")

# ───────── helper: waveform→examples ─────────

def get_waveform_to_examples():
    try:
        from torch_vggish_yamnet.input_proc import waveform_to_examples
        print("Using waveform_to_examples from torch_vggish_yamnet"); return waveform_to_examples
    except ImportError: pass
    try:
        from torch_vggish_yamnet.input_proc import wavfile_to_examples
        print("Using wavfile_to_examples (older wheel)");      return wavfile_to_examples
    except ImportError: pass

    # DIY – identical framing params as TF YAMNet
    print("  Using DIY log‑mel extraction (slower)")
    import torch.nn.functional as F
    from torchaudio.transforms import MelSpectrogram
    mel = MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=160, win_length=400,
        n_mels=64, f_min=125.0, f_max=7500.0, power=2.0)

    def diy(wave:np.ndarray, sr:int=16000):
        if sr!=16000:
            raise ValueError("DIY expects 16 kHz audio")
        spec = mel(torch.tensor(wave).unsqueeze(0))            # (1,64,T)
        spec = torch.nan_to_num(spec, nan=0.0)                 # remove any NaN from mel
        spec = (spec + 1e-6).log().squeeze(0)                # (64,T)
        frames, hop = 96, 48
        if spec.shape[1] < frames:
            spec = F.pad(spec, (0, frames-spec.shape[1]))
        num = 1 + (spec.shape[1] - frames)//hop
        out = torch.zeros(num,1,64,96)
        for i in range(num):
            out[i,0] = spec[:, i*hop : i*hop+frames]
        return out.numpy()
    return diy

waveform_to_examples = get_waveform_to_examples()

# ───────── predict one file ─────────

def predict_one(wav_path:str):
    print(f"\n🎧  {os.path.basename(wav_path)}")
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0]>1:
        wav = wav.mean(0, keepdim=True)
    if sr!=16000:
        wav = torchaudio.transforms.Resample(sr,16000)(wav); sr=16000

    examples = torch.from_numpy(
        waveform_to_examples(wav.squeeze().numpy().astype(np.float32), sr)
    )                                                # (N,1,64,96)
    print(f"   patches     {examples.shape}  (N,1,64,96)")

    with torch.no_grad():
        embeds, *_ = yamnet(examples)                # (N,1024,1,1) –> squeeze later
        embeds = embeds.squeeze(-1).squeeze(-1)      # (N,1024)
        mean_embed = embeds.mean(0, keepdim=True)    # (1,1024)
        if torch.isnan(mean_embed).any():
            print("    mean_embed has NaNs – skipping"); return
        logits = mlp(mean_embed).squeeze()           # (59,) already post‑sigmoid
        probs  = logits.clamp(0,1).numpy()           # force [0,1] just in case

    topk = probs.argsort()[-5:][::-1]
    for idx in topk:
        print(f"      ▸ {label_names[idx]:<25} {probs[idx]:.2f}")

# ───────── batch iterate ─────────
for fn in sorted(os.listdir(audio_dir)):
    if fn.lower().endswith('.wav'):
        try:
            predict_one(os.path.join(audio_dir,fn))
        except Exception as e:
            print(f"    ERROR on {fn}: {e}")