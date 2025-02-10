import torch
import torch.nn as nn


class WAIRDOriginal(nn.Module):
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.__mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, sequence: torch.Tensor):
        idx_to_keep = [0, 1] + [
            sequence.shape[2] * i + j
            for i in range(sequence.shape[1])
            for j in range(2, sequence.shape[2])
        ]
        
        out = self.__mlp(sequence.flatten(start_dim=1)[:, idx_to_keep])
        return out
