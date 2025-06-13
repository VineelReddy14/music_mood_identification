"""
yamnet_mlp_infer.py
───────────────────
Step-2: YAMNet (offline, Google weights) + your 59-tag MLP head.

CLI usage
---------
# print top-5 mood tags
python yamnet_mlp_infer.py audio.wav --mood -k 5

# just show embedding shape (default)
python yamnet_mlp_infer.py audio.wav
"""

# ───────────── Imports ─────────────
from pathlib import Path
import argparse, textwrap
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, torchaudio.transforms as TA

# ─────────── Hyper-parameters ───────────
class CommonParams:
    TARGET_SAMPLE_RATE           = 16_000
    STFT_WINDOW_LENGTH_SECONDS   = 0.025
    STFT_HOP_LENGTH_SECONDS      = 0.010
    NUM_MEL_BANDS                = 64
    MEL_MIN_HZ, MEL_MAX_HZ       = 125, 7_500
    LOG_OFFSET                   = 1e-3
    PATCH_WINDOW_IN_SECONDS      = 0.96        # 96 frames ×10 ms

class YAMNetParams:
    NUM_CLASSES       = 521
    BATCHNORM_EPSILON = 1e-4

# ─────────── Pre-processing ────────────
class _LogMelSpec(TA.MelSpectrogram):
    def forward(self, wav):
        spec = self.spectrogram(wav) ** 0.5
        mel  = self.mel_scale(spec)
        return torch.log(mel + CommonParams.LOG_OFFSET)

class WaveformToInput(nn.Module):
    """Waveform (C,T) → [N,1,96,64] patches."""
    def __init__(self):
        super().__init__()
        sr   = CommonParams.TARGET_SAMPLE_RATE
        win  = int(sr * CommonParams.STFT_WINDOW_LENGTH_SECONDS)
        hop  = int(sr * CommonParams.STFT_HOP_LENGTH_SECONDS)
        fft  = 2 ** int(np.ceil(np.log2(win)))
        self.mel = _LogMelSpec(sr, n_fft=fft, win_length=win, hop_length=hop,
                               n_mels=CommonParams.NUM_MEL_BANDS,
                               f_min=CommonParams.MEL_MIN_HZ,
                               f_max=CommonParams.MEL_MAX_HZ)

    def forward(self, wav, sr):
        x = wav.mean(0, keepdim=True)
        if sr != CommonParams.TARGET_SAMPLE_RATE:
            x = TA.Resample(sr, CommonParams.TARGET_SAMPLE_RATE)(x)
        x = self.mel(x).squeeze(0).T                      # [T,64]

        win_frames = int(CommonParams.PATCH_WINDOW_IN_SECONDS /
                         CommonParams.STFT_HOP_LENGTH_SECONDS)
        n_chunks   = x.shape[0] // win_frames
        x = x[: n_chunks * win_frames]
        return x.reshape(n_chunks, 1, win_frames, x.shape[1]).float()

# ───────── YAMNet blocks (matching checkpoint names) ─────────
class Conv2d_tf(nn.Conv2d):
    """TF-style SAME padding reproduced in PyTorch."""
    def __init__(self,*a,**k): k.pop("padding",None); super().__init__(*a,**k)
    def _same_padding(self, inp, dim):
        in_sz  = inp.size(dim+2)
        k,d,s  = self.kernel_size[dim], self.dilation[dim], self.stride[dim]
        eff_k  = (k-1)*d + 1
        out_sz = (in_sz + s - 1)//s
        pad    = max(0, (out_sz-1)*s + eff_k - in_sz)
        return int(pad % 2), pad
    def forward(self, x):
        o1,p1 = self._same_padding(x,0); o2,p2 = self._same_padding(x,1)
        if o1 or o2: x = F.pad(x, [0,o2,0,o1])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        [p1//2, p2//2], self.dilation, self.groups)

class CBR(nn.Module):
    """Conv → BatchNorm → ReLU (keeps explicit attr names)."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn   = nn.BatchNorm2d(conv.out_channels,
                                   eps=YAMNetParams.BATCHNORM_EPSILON)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class Conv(nn.Module):
    """First regular conv block."""
    def __init__(self,k,s,cin,cout):
        super().__init__()
        self.fused = CBR(Conv2d_tf(cin, cout, kernel_size=k, stride=s,
                                   bias=False))
    def forward(self,x): return self.fused(x)

class SepConv(nn.Module):
    """Depthwise-separable conv block (depthwise + pointwise)."""
    def __init__(self,k,s,cin,cout):
        super().__init__()
        self.depthwise_conv = CBR(
            Conv2d_tf(cin, cin, kernel_size=k, stride=s,
                      groups=cin, bias=False)
        )
        self.pointwise_conv = CBR(
            Conv2d_tf(cin, cout, kernel_size=1, stride=1, bias=False)
        )
    def forward(self,x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)

class YAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(Conv,[3,3],2,32),  (SepConv,[3,3],1,64),
               (SepConv,[3,3],2,128),(SepConv,[3,3],1,128),
               (SepConv,[3,3],2,256),(SepConv,[3,3],1,256),
               (SepConv,[3,3],2,512)] + [(SepConv,[3,3],1,512)]*5 + \
              [(SepConv,[3,3],2,1024),(SepConv,[3,3],1,1024)]
        cin=1; self.layer_names=[]
        for i,(blk,k,s,cout) in enumerate(cfg,1):
            name=f"layer{i}"; setattr(self,name,blk(k,s,cin,cout))
            self.layer_names.append(name); cin=cout
        self.classifier = nn.Linear(cin, YAMNetParams.NUM_CLASSES)
    def forward(self,x,to_prob=False):
        for n in self.layer_names: x=getattr(self,n)(x)
        x = F.adaptive_avg_pool2d(x,1)
        emb = x.clone()
        logits = self.classifier(x.view(x.size(0),-1))
        return emb, torch.sigmoid(logits) if to_prob else logits

# ────────── Custom 59-tag MLP head ──────────
class MLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=59,
                 d1=0.3, d2=0.3, d3=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim,512), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(512,256),    nn.ReLU(), nn.Dropout(d2),
            nn.Linear(256,128),    nn.ReLU(), nn.Dropout(d3),
            nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self,x): return self.model(x)

# ────────── Local-file helpers ──────────
ROOT  = Path(__file__).resolve().parent
W_YAM = ROOT / "yamnet_weights.pth"
W_MLP = ROOT / "model_pytorch_model_v1_fold1.pt"
F_TAG = ROOT / "tag_index_mapping.txt"

def load_local_yamnet(w=W_YAM):
    if not Path(w).exists(): raise FileNotFoundError(f"Missing {w}")
    m = YAMNet(); m.load_state_dict(torch.load(w, map_location="cpu"))
    return m

def load_local_mlp(w=W_MLP):
    if not Path(w).exists(): raise FileNotFoundError(f"Missing {w}")
    mlp = MLP(); mlp.load_state_dict(torch.load(w, map_location="cpu"))
    return mlp

def load_tag_names(f=F_TAG):
    if not Path(f).exists(): raise FileNotFoundError(f"Missing {f}")
    with open(f) as fh: tags=[ln.strip().split()[-1] for ln in fh]
    return tags

# ────────── End-to-end prediction ──────────
def predict_mood(audio_path, top_k=10, device="cpu"):
    wav, sr = torchaudio.load(audio_path)
    patches = WaveformToInput()(wav, sr).to(device)

    yam = load_local_yamnet().to(device).eval()
    mlp = load_local_mlp().to(device).eval()
    tags = load_tag_names()

    with torch.no_grad():
        emb,_  = yam(patches)
        emb    = emb.squeeze(-1).squeeze(-1).mean(0, keepdim=True)  # [1,1024]
        probs  = mlp(emb).squeeze(0).cpu().numpy()                  # [59]

    idx = probs.argsort()[::-1][:top_k]
    return [(tags[i], float(probs[i])) for i in idx]

# ────────── CLI interface ──────────
if __name__ == "__main__":
    desc = ("Default (no flags) shows embedding shape.\n"
            "--mood prints top-k mood tags.")
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(desc))
    ap.add_argument("audio", help="Path to audio file")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--mood", action="store_true", help="print mood tags")
    mode.add_argument("--emb",  action="store_true",
                      help="just embedding shape (default)")
    ap.add_argument("-k","--topk", type=int, default=10, help="top-k tags")
    ap.add_argument("--cuda", action="store_true", help="use CUDA if available")
    args = ap.parse_args()

    DEVICE = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.mood:
        for tag, p in predict_mood(args.audio, args.topk, DEVICE):
            print(f"{tag:25} {p:.3f}")
    else:   # embedding info
        wav, sr = torchaudio.load(args.audio)
        patches = WaveformToInput()(wav, sr)
        with torch.no_grad():
            emb,_ = load_local_yamnet().eval()(patches)
        emb = emb.squeeze(-1).squeeze(-1).numpy()
        print("Embeddings shape:", emb.shape)
