import torch
import torch.nn as nn
from pl_bolts.models.vision.unet import UNet


class MLPUNet(nn.Module):
    
    def __init__(
        self, num_classes: int, mlp_input_dim: int,
        u_input_channels: int, u_num_layers: int, u_features_start: int
    ):
        super().__init__()
        self.unet = UNet(num_classes, u_input_channels, u_num_layers, u_features_start)
        
        # Mixing MLP and UNet outputs
        feats = u_features_start * 2 ** (u_num_layers - 1)
        self.mix_layer = nn.Conv2d(feats + 256, feats, kernel_size=1)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
    
    def forward(self, image, sequence):
        xi = [self.unet.layers[0](image)]
        # Down path
        for layer in self.unet.layers[1: self.unet.num_layers]:
            xi.append(layer(xi[-1]))
        
        idx_to_keep = [0, 1] + [
            sequence.shape[2] * i + j
            for i in range(sequence.shape[1])
            for j in range(2, sequence.shape[2])
        ]
        
        mlp_embedding = self.mlp(sequence.flatten(start_dim=1)[:, idx_to_keep])
        
        # Merging
        mlp_embedding = mlp_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, xi[-1].shape[-2], xi[-1].shape[-1])
        merged_embedding = torch.cat((xi[-1], mlp_embedding), dim=1)
        xi[-1] = self.mix_layer(merged_embedding)
        
        # Up path
        for i, layer in enumerate(self.unet.layers[self.unet.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        return self.unet.layers[-1](xi[-1])
