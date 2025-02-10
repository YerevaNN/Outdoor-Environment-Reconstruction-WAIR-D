from typing import Optional

import torch.nn as nn
from mmseg.models import MultiLevelNeck

from src.networks import CLIPPlus
from src.networks.step_neck import StepNeck
from src.networks.upernet import FPN_fuse, PSPModule


class CLIPPlusUPerNet(nn.Module):
    
    def __init__(
        self, num_classes: int, image_size: int, mlp_input_dim: int, min_mlp_tokens: int, num_channels: int,
        v_num_channels: int, v_patch_size: int,
        v_hidden_size: int, v_num_hidden_layers: int, v_num_attention_heads: int,
        mixer_out: int, res_hidden_states: list[int], up_pool_scales: list[int],
        neck_scales: list[float], neck_size: list[int],
        pretrained: str, reconstruction: bool
    ):
        super().__init__()
        
        assert (
            mixer_out is None or int(mixer_out ** 0.5) ** 2 == mixer_out,
            "Mixer's output size should be a square of a whole number"
        )
        
        self.v_hidden_size = v_hidden_size
        self.v_patch_size = v_patch_size
        
        self.reconstruction = reconstruction
        if reconstruction:
            self.conv = nn.Conv2d(num_channels, v_num_channels, kernel_size=3, padding="same")
        else:
            self.conv = lambda x: x
        
        self.clip_plus = CLIPPlus(
            mlp_input_dim=mlp_input_dim, v_num_channels=v_num_channels, v_patch_size=v_patch_size,
            v_hidden_size=v_hidden_size, v_num_hidden_layers=v_num_hidden_layers,
            v_num_attention_heads=v_num_attention_heads, pretrained=pretrained,
            reconstruction=reconstruction
        )
        self.res_hidden_states = res_hidden_states
        feature_channels = [v_hidden_size] * (v_num_hidden_layers + 2)
        if res_hidden_states is not None:
            feature_channels = [feature_channels[i] for i in res_hidden_states]
        
        self.num_tokes = (image_size // v_patch_size) ** 2
        self.mixer_out = mixer_out or self.num_tokes
        self.mixers = nn.ModuleList(
            [
                nn.Conv1d(self.num_tokes + min_mlp_tokens + 1, self.mixer_out, 1) for _ in range(len(feature_channels))
            ]
        )
        
        if neck_scales:
            if len(neck_size) == 1:
                self.neck = MultiLevelNeck(
                    in_channels=[v_hidden_size] * len(feature_channels),
                    out_channels=neck_size[0],
                    scales=neck_scales
                )
                feature_channels = [neck_size[0]] * len(feature_channels)
            else:
                self.neck = StepNeck(
                    in_channels=[v_hidden_size] * len(feature_channels),
                    out_channels=neck_size,
                    scales=neck_scales
                )
                feature_channels = neck_size
        else:
            self.neck = None
        
        self.PPN = PSPModule(feature_channels[-1], bin_sizes=up_pool_scales)
        self.FPN = FPN_fuse(feature_channels)
        if feature_channels[0] // (v_patch_size ** 2) == feature_channels[0] / (v_patch_size ** 2):
            head_out = feature_channels[0] // (v_patch_size ** 2)
        else:
            head_out = feature_channels[0] // self.num_tokes
        
        self.head = nn.Conv2d(
            feature_channels[0] if self.neck else head_out,
            num_classes, kernel_size=3, padding=1
        )
    
    def forward(
        self, image, sequence,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        clip_out = self.clip_plus(
            self.conv(image), sequence,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        clip_hidden_states = clip_out.hidden_states
        
        h = w = int(self.mixer_out ** 0.5)
        
        j = 0
        embeddings = []
        for i, v in enumerate(clip_hidden_states + (clip_out.last_hidden_state,)):
            if self.res_hidden_states and i not in self.res_hidden_states:
                continue
            m = self.mixers[j]
            j += 1
            embeddings.append(m(v[:, :m.in_channels]).reshape(-1, h, w, self.v_hidden_size).permute(0, 3, 1, 2))
        
        if self.neck:
            embeddings = list(self.neck(embeddings))
        # Up path
        embeddings[-1] = self.PPN(embeddings[-1])
        output = self.FPN(embeddings)
        
        if not self.neck:
            h = w = int(self.num_tokes ** 0.5)
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
