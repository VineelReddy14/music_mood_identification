# Pytorch_model.py

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout1=0.3, dropout2=0.3, dropout3=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Binary multi-label
        )

    def forward(self, x):
        return self.model(x)