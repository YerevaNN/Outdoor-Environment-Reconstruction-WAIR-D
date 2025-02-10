import torch
import torch.nn as nn
from pl_bolts.models.vision.unet import DoubleConv, Down

from src.networks.upernet import FPN_fuse, PSPModule


class MLPUNetUPerNet(nn.Module):
    
    def __init__(
        self, num_classes: int, mlp_input_dim: int,
        u_input_channels: int, u_num_layers: int, u_features_start: int, up_pool_scales: list[int]
    ):
        super().__init__()
        
        self.u_num_layers = u_num_layers
        layers = [DoubleConv(u_input_channels, u_features_start)]
        
        feats = u_features_start
        for _ in range(u_num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        
        self.unet_encoder = nn.ModuleList(layers)
        
        # Mixing MLP and UNet outputs
        self.mix_layer = nn.Conv2d(feats + 256, feats, kernel_size=1)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
        
        feature_channels = [u_features_start * 2 ** i for i in range(u_num_layers)]
        self.PPN = PSPModule(feature_channels[-1], bin_sizes=up_pool_scales)
        self.FPN = FPN_fuse(feature_channels)
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)
    
    def forward(self, image, sequence):
        xi = [self.unet_encoder[0](image)]
        # Down path
        for layer in self.unet_encoder[1: self.u_num_layers]:
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
        xi[-1] = self.PPN(xi[-1])
        return self.head(self.FPN(xi))
