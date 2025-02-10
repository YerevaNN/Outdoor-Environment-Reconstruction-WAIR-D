import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel, ViTConfig, ViTModel

from src.networks.upernet import FPN_fuse, PSPModule


class MLPViTUPerNet(nn.Module):
    
    def __init__(
        self, num_classes: int, mlp_input_dim: int,
        v_num_channels: int, v_patch_size: int,
        v_hidden_size: int, v_num_hidden_layers: int, v_num_attention_heads: int,
        res_hidden_states: list[int], up_pool_scales: list[int], pretrained: str
    ):
        super().__init__()
        
        assert v_hidden_size % (v_patch_size ** 2) == 0, "v_hidden_size must be divisible by v_patch_size^2"
        
        vit_config = dict(
            num_channels=v_num_channels, patch_size=v_patch_size,
            hidden_size=v_hidden_size, num_hidden_layers=v_num_hidden_layers,
            num_attention_heads=v_num_attention_heads, output_hidden_states=True
        )
        self.v_hidden_size = v_hidden_size
        self.v_patch_size = v_patch_size
        self.vit = (
            CLIPVisionModel(CLIPVisionConfig(**vit_config)).from_pretrained(
                pretrained, output_hidden_states=True
            ) if pretrained
            else ViTModel(ViTConfig(**vit_config), add_pooling_layer=False)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, v_patch_size ** 2),
        )
        
        self.res_hidden_states = res_hidden_states
        feature_channels = [v_hidden_size + v_patch_size ** 2] * (v_num_hidden_layers + 2)
        if res_hidden_states is not None:
            feature_channels = [feature_channels[i] for i in res_hidden_states]
        self.PPN = PSPModule(feature_channels[-1], bin_sizes=up_pool_scales)
        self.FPN = FPN_fuse(feature_channels)
        self.head = nn.Conv2d(feature_channels[0] // (v_patch_size ** 2), num_classes, kernel_size=3, padding=1)
    
    def forward(self, image, sequence):
        vit_out = self.vit(image)
        vit_hidden_states = vit_out.hidden_states
        
        idx_to_keep = [0, 1] + [
            sequence.shape[2] * i + j
            for i in range(sequence.shape[1])
            for j in range(2, sequence.shape[2])
        ]
        
        mlp_embedding = self.mlp(sequence.flatten(start_dim=1)[:, idx_to_keep])
        
        # Merging
        h = w = int((vit_out.last_hidden_state.shape[1] - 1) ** 0.5)
        mlp_embedding = mlp_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        merged_embeddings = [
            torch.cat((v[:, 1:].reshape(-1, h, w, self.v_hidden_size).permute(0, 3, 1, 2), mlp_embedding), dim=1)
            for i, v in enumerate(vit_hidden_states + (vit_out.last_hidden_state,))
            if self.res_hidden_states is None or i in self.res_hidden_states
        ]
        
        # Up path
        merged_embeddings[-1] = self.PPN(merged_embeddings[-1])
        output = self.FPN(merged_embeddings)
        
        # unpatching the output
        # the next line of code was thoroughly thought and tested, never to be touched again
        output = output.permute(0, 2, 3, 1).reshape(
            -1, h, w, self.head.in_channels, self.v_patch_size, self.v_patch_size
        ).permute(
            0, 3, 1, 4, 2, 5
        ).reshape(
            -1, self.head.in_channels, self.v_patch_size * h, self.v_patch_size * w
        )
        
        return self.head(output)
