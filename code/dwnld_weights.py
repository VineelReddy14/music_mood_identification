import torch, urllib.request, pathlib

CKPT_URL = "https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth"
dst = pathlib.Path("/mnt/data/Vineel/Music_mood_identification/yamnet_weights.pth")

if not dst.exists():
    print("Downloading YAMNet pretrained weights …")
    urllib.request.urlretrieve(CKPT_URL, dst)
    print("Saved to", dst)
