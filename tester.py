# demo_predict.py
from pathlib import Path
from yamnet_mlp_infer import predict_timeline_and_theme   # or predict_mood

AUDIO = Path("/mnt/data/Vineel/epidemics_audios/ES_Rocktronic_Def_Lev.wav")

# device can be "cpu" or "cuda"
timeline, theme = predict_timeline_and_theme(
    audio_path=AUDIO,
    top_k=3,
    device="cpu"
)

print("Whole-clip theme (top-3):")
for tag, prob in theme:
    print(f"{tag:25} {prob:.3f}")

print("\nTimeline (one entry per second):")
for t, tags in timeline[:10]:          # show first 10 seconds
    tag_str = ", ".join(f"{n}:{p:.2f}" for n, p in tags)
    print(f"{t:6.2f}s  {tag_str}")
