from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class CLIPPlus(nn.Module):
    
    def __init__(
        self, mlp_input_dim: int, v_num_channels: int, v_patch_size: int,
        v_hidden_size: int, v_num_hidden_layers: int, v_num_attention_heads: int, pretrained: str,
        reconstruction: bool
    ):
        super().__init__()
        
        self.v_hidden_size = v_hidden_size
        self.v_patch_size = v_patch_size
        self.reconstruction = reconstruction
        
        self.clip = CLIPVisionModel(
            CLIPVisionConfig(
                num_channels=v_num_channels, patch_size=v_patch_size,
                hidden_size=v_hidden_size, num_hidden_layers=v_num_hidden_layers,
                num_attention_heads=v_num_attention_heads, output_hidden_states=True
            )
        )
        if pretrained:
            self.clip = self.clip.from_pretrained(pretrained, output_hidden_states=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, v_hidden_size),
        )
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        sequence: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions or self.clip.config.output_attentions
        output_hidden_states = output_hidden_states or self.clip.config.output_hidden_states
        return_dict = return_dict or self.clip.config.use_return_dict
        
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        if not self.reconstruction:
            idx_to_keep = [0, 1] + [
                sequence.shape[2] * i + j
                for i in range(sequence.shape[1])
                for j in range(2, sequence.shape[2])
            ]
            sequence = sequence.flatten(start_dim=1)[:, idx_to_keep]
            sequence_embedding = self.mlp(sequence)[:, None]
        else:
            sequence_embedding = self.mlp(sequence)
        
        img_embeddings = self.clip.vision_model.embeddings(pixel_values)
        img_embeddings = self.clip.vision_model.pre_layrnorm(img_embeddings)
        
        encoder_outputs = self.clip.vision_model.encoder(
            inputs_embeds=torch.cat([img_embeddings, sequence_embedding], dim=1),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.clip.vision_model.post_layernorm(pooled_output)
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
