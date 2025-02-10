import torch
from pl_bolts.models.vision.unet import DoubleConv, Down
from torch import nn


class MLPUNetEncoder(nn.Module):
    
    def __init__(self, mlp_input_dim: int, u_input_channels: int, u_num_layers: int, u_features_start: int):
        if u_num_layers < 1:
            raise ValueError(f"num_layers = {u_num_layers}, expected: num_layers > 0")
        
        super().__init__()
        self.num_layers = u_num_layers
        
        layers = [DoubleConv(u_input_channels, u_features_start)]
        
        feats = u_features_start
        for _ in range(u_num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        
        self.unet_encoder = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(feats + 256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, image, sequence):
        unet_embedding = self.unet_encoder(image)
        
        idx_to_keep = [0, 1] + [
            sequence.shape[2] * i + j
            for i in range(sequence.shape[1])
            for j in range(2, sequence.shape[2])
        ]
        mlp_embedding = self.mlp(sequence.flatten(start_dim=1)[:, idx_to_keep])
        
        return self.output_layer(torch.cat((unet_embedding, mlp_embedding), dim=1))
