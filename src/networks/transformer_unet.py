from typing import Optional

import torch
import torch.nn as nn
from pl_bolts.models.vision.unet import UNet
from torch.nn import init


class TransformerUNet(nn.Module):
    
    def __init__(
        self, num_classes: int,
        t_input_dim: int, t_embed_dim: int, t_num_heads: int, t_num_layers: int,
        u_input_channels: int, u_num_layers: int, u_features_start: int, u_input_size: Optional[int]
    ):
        super().__init__()
        self.unet = UNet(num_classes, u_input_channels, u_num_layers, u_features_start)
        
        # Mixing transformer and UNet outputs
        feats = u_features_start * 2 ** (u_num_layers - 1)
        self.mix_layer = nn.Conv2d(feats + t_embed_dim, feats, kernel_size=1)
        
        # Embedding layer (no positional embeddings or tokenization)
        self.t_embedding = nn.Linear(t_input_dim, t_embed_dim, bias=False)
        init.xavier_uniform_(self.t_embedding.weight)
        
        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=t_embed_dim,
            nhead=t_num_heads
        )
        self.t_encoder = nn.TransformerEncoder(t_encoder_layer, num_layers=t_num_layers)
        
        self.u_input_size = u_input_size
        
        if self.u_input_size is not None:
            u_embed_size = (u_input_size // 2 ** (u_num_layers - 1)) ** 2
            self.t_conv = nn.Conv1d(in_channels=u_input_channels // 2, out_channels=u_embed_size, kernel_size=1)
    
    def forward(self, image, sequence, mask=None):
        xi = [self.unet.layers[0](image)]
        # Down path
        for layer in self.unet.layers[1: self.unet.num_layers]:
            xi.append(layer(xi[-1]))
        
        # Transformer
        input_embedding = self.t_embedding(sequence)
        output_embedding = self.t_encoder(input_embedding, src_key_padding_mask=mask and mask.T)
        # The output embeddings for masked tokens are NaN
        output_embedding = output_embedding.nan_to_num_()
        
        if self.u_input_size is None:
            # Calculate the average of the output embeddings along the sequence dimension
            # Averaging only not masked output embeddings
            eps = 1e-7
            if mask is not None:
                avg_embedding = output_embedding.sum(axis=1) / ((~mask).sum(axis=1).unsqueeze(1) + eps)
            else:
                avg_embedding = output_embedding.mean(axis=1)
            
            # Merging
            t_embedding = avg_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, xi[-1].shape[-2], xi[-1].shape[-1])
        else:
            t_embedding = self.t_conv(output_embedding).reshape(
                -1, xi[-1].shape[-2], xi[-1].shape[-1], output_embedding.shape[-1]
            ).permute(0, 3, 1, 2)
        merged_embedding = torch.cat((xi[-1], t_embedding), dim=1)
        
        xi[-1] = self.mix_layer(merged_embedding)
        
        # Up path
        for i, layer in enumerate(self.unet.layers[self.unet.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        return self.unet.layers[-1](xi[-1])
