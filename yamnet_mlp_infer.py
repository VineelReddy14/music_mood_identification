"""
yamnet_mlp_infer.py  –  Step-3
• Offline YAMNet + 59-tag MLP
• Per-second timeline predictions  (≈ 0.96 s resolution)
• Whole-clip theme via mean-pooled embedding

CLI
----
# whole-clip top-5 tags
python yamnet_mlp_infer.py song.wav --mood -k 5

# timeline (one line per second)
python yamnet_mlp_infer.py song.wav --timeline -k 3
"""

# ───────── Imports ─────────
from pathlib import Path
import argparse, textwrap, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, torchaudio.transforms as TA

# ───────── Hyper-params ─────────
class CommonParams:
    TARGET_SAMPLE_RATE         = 16_000
    STFT_WINDOW_LEN_SEC        = 0.025
    STFT_HOP_LEN_SEC           = 0.010
    NUM_MEL_BANDS              = 64
    MEL_MIN_HZ, MEL_MAX_HZ     = 125, 7_500
    LOG_OFFSET                 = 1e-3
    PATCH_WINDOW_SEC           = 0.96      # 96 frames ×10 ms

class YAMNetParams:
    NUM_CLASSES       = 521
    BATCHNORM_EPSILON = 1e-4

# ───────── Pre-processing ─────────
class _LogMel(TA.MelSpectrogram):
    def forward(self, wav):
        spec = self.spectrogram(wav) ** 0.5
        mel  = self.mel_scale(spec)
        return torch.log(mel + CommonParams.LOG_OFFSET)

class WaveformToInput(nn.Module):
    """Waveform → [N,1,96,64] (≈ 1 s patches)."""
    def __init__(self):
        super().__init__()
        sr   = CommonParams.TARGET_SAMPLE_RATE
        win  = int(sr * CommonParams.STFT_WINDOW_LEN_SEC)
        hop  = int(sr * CommonParams.STFT_HOP_LEN_SEC)
        fft  = 2 ** int(np.ceil(np.log2(win)))
        self.mel = _LogMel(sr, n_fft=fft, win_length=win, hop_length=hop,
                           n_mels=CommonParams.NUM_MEL_BANDS,
                           f_min=CommonParams.MEL_MIN_HZ,
                           f_max=CommonParams.MEL_MAX_HZ)

    def forward(self, wav, sr):
        x = wav.mean(0, keepdim=True)
        if sr != CommonParams.TARGET_SAMPLE_RATE:
            x = TA.Resample(sr, CommonParams.TARGET_SAMPLE_RATE)(x)
        x  = self.mel(x).squeeze(0).T                       # [T,64]

        win_frames = int(CommonParams.PATCH_WINDOW_SEC /
                         CommonParams.STFT_HOP_LEN_SEC)     # 96
        n_chunks   = x.shape[0] // win_frames
        x = x[: n_chunks * win_frames]
        return x.reshape(n_chunks, 1, win_frames, x.shape[1]).float()

# ───────── YAMNet (matching weight keys) ─────────
class Conv2d_tf(nn.Conv2d):
    def __init__(self,*a,**k): k.pop("padding",None); super().__init__(*a,**k)
    def _same(self,x,d):
        in_sz = x.size(d+2); k,dil,s = self.kernel_size[d],self.dilation[d],self.stride[d]
        eff   = (k-1)*dil + 1
        out   = (in_sz + s - 1)//s
        pad   = max(0,(out-1)*s + eff - in_sz)
        return int(pad%2), pad
    def forward(self,x):
        o1,p1 = self._same(x,0); o2,p2 = self._same(x,1)
        if o1 or o2: x = F.pad(x,[0,o2,0,o1])
        return F.conv2d(x,self.weight,self.bias,self.stride,[p1//2,p2//2],
                        self.dilation,self.groups)

class CBR(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn   = nn.BatchNorm2d(conv.out_channels, eps=YAMNetParams.BATCHNORM_EPSILON)
        self.relu = nn.ReLU()
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class Conv(nn.Module):
    def __init__(self,k,s,cin,cout):
        super().__init__()
        self.fused = CBR(Conv2d_tf(cin, cout, k, s, bias=False))
    def forward(self,x): return self.fused(x)

class SepConv(nn.Module):
    def __init__(self,k,s,cin,cout):
        super().__init__()
        self.depthwise_conv = CBR(Conv2d_tf(cin, cin, k, s, groups=cin, bias=False))
        self.pointwise_conv = CBR(Conv2d_tf(cin, cout, 1, 1, bias=False))
    def forward(self,x): return self.pointwise_conv(self.depthwise_conv(x))

class YAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        cfg=[(Conv,[3,3],2,32),(SepConv,[3,3],1,64),
             (SepConv,[3,3],2,128),(SepConv,[3,3],1,128),
             (SepConv,[3,3],2,256),(SepConv,[3,3],1,256),
             (SepConv,[3,3],2,512)]+[(SepConv,[3,3],1,512)]*5+[
             (SepConv,[3,3],2,1024),(SepConv,[3,3],1,1024)]
        cin=1; self.layer_names=[]
        for i,(blk,k,s,cout) in enumerate(cfg,1):
            name=f"layer{i}"; setattr(self,name,blk(k,s,cin,cout))
            self.layer_names.append(name); cin=cout
        self.classifier = nn.Linear(cin, YAMNetParams.NUM_CLASSES)
    def forward(self,x,to_prob=False):
        for n in self.layer_names: x=getattr(self,n)(x)
        x=F.adaptive_avg_pool2d(x,1)
        emb=x.clone()
        logits=self.classifier(x.view(x.size(0),-1))
        return emb, torch.sigmoid(logits) if to_prob else logits

# ───────── 59-tag MLP head ─────────
class MLP(nn.Module):
    def __init__(self,in_dim=1024,out_dim=59,d1=0.3,d2=0.3,d3=0.2):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(in_dim,512), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(512,256),    nn.ReLU(), nn.Dropout(d2),
            nn.Linear(256,128),    nn.ReLU(), nn.Dropout(d3),
            nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self,x): return self.model(x)

# ───────── Local helpers ─────────
ROOT  = Path(__file__).resolve().parent
W_YAM = ROOT/"yamnet_weights.pth"
W_MLP = ROOT/"model_pytorch_model_v1_fold1.pt"
F_TAG = ROOT/"tag_index_mapping.txt"

def load_local_yamnet(): m=YAMNet(); m.load_state_dict(torch.load(W_YAM,"cpu")); return m
def load_local_mlp():    mlp=MLP();  mlp.load_state_dict(torch.load(W_MLP,"cpu"));return mlp
def load_tags(): return [ln.strip().split()[-1] for ln in open(F_TAG)]

# ───────── Core inference helpers ─────────
def _per_patch_probs(patches, yam, mlp):
    """patches [N,1,96,64] → probs [N,59] on CPU/GPU of yam."""
    with torch.no_grad():
        emb,_ = yam(patches)                         # [N,1024,1,1]
        emb   = emb.squeeze(-1).squeeze(-1)          # [N,1024]
        probs = mlp(emb)                             # [N,59]
    return probs

def predict_timeline_and_theme(audio_path, top_k=3, device="cpu"):
    """Returns (timeline list, theme list)
       timeline: list[(sec, [(tag,prob),...])]   for each 0.96-s patch
       theme:    [(tag,prob), ...]  from mean-pooled embedding
    """
    wav, sr = torchaudio.load(audio_path)
    patches = WaveformToInput()(wav, sr).to(device)   # [N,1,96,64]
    N = patches.size(0)
    yam  = load_local_yamnet().to(device).eval()
    mlp  = load_local_mlp().to(device).eval()
    tags = load_tags()

    # per-patch probabilities
    probs = _per_patch_probs(patches, yam, mlp).cpu().numpy()   # [N,59]

    timeline=[]
    for i in range(N):
        sec = round(i * CommonParams.PATCH_WINDOW_SEC, 2)       # start-time
        idx = probs[i].argsort()[::-1][:top_k]
        timeline.append((sec, [(tags[j], float(probs[i][j])) for j in idx]))

    # whole-clip theme via mean-pooled embedding
    with torch.no_grad():
        emb,_=yam(patches)
        emb  = emb.squeeze(-1).squeeze(-1).mean(0, keepdim=True)  # [1,1024]
        theme_probs = mlp(emb).squeeze(0).cpu().numpy()
    idx = theme_probs.argsort()[::-1][:top_k]
    theme=[(tags[j], float(theme_probs[j])) for j in idx]

    return timeline, theme

# ───────── CLI ─────────
if __name__ == "__main__":
    desc=("No flag: embedding shape\n"
          "--mood    : whole-clip theme\n"
          "--timeline: per-second mood timeline")
    ap=argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                               description=textwrap.dedent(desc))
    ap.add_argument("audio")
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--mood", action="store_true")
    g.add_argument("--timeline", action="store_true")
    ap.add_argument("-k","--topk", type=int, default=5)
    ap.add_argument("--cuda", action="store_true")
    args=ap.parse_args()

    DEVICE="cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.timeline:
        tl, theme = predict_timeline_and_theme(args.audio, args.topk, DEVICE)
        print("Whole-clip theme:")
        for tag,p in theme: print(f"  {tag:25} {p:.3f}")
        print("\nTimeline (every ~1 s):")
        for sec, lst in tl:
            tags_str = ", ".join([f"{t}:{p:.2f}" for t,p in lst])
            print(f"{sec:6.2f}s  {tags_str}")

    elif args.mood:
        _, theme = predict_timeline_and_theme(args.audio, args.topk, DEVICE)
        for tag,p in theme: print(f"{tag:25} {p:.3f}")

    else:
        wav,sr=torchaudio.load(args.audio)
        patches=WaveformToInput()(wav,sr)
        with torch.no_grad(): emb,_=load_local_yamnet().eval()(patches)
        print("Embeddings shape:", emb.squeeze(-1).squeeze(-1).shape)
