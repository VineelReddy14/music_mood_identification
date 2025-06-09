import torch
from torch_vggish_yamnet import yamnet

# Load pretrained YAMNet model
model = yamnet.yamnet(pretrained=True)
model.eval()

# Save weights to the desired location
save_path = "/mnt/data/Vineel/jamendo_project/models/yamnet_pytorch_weights.pth"
torch.save(model.state_dict(), save_path)

print(f"YAMNet weights saved at: {save_path}")
