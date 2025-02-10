import torch.nn as nn
from torch.nn import init


class Transformer(nn.Module):
    
    def __init__(
        self, input_dim: int, embed_dim: int, num_heads: int, num_layers: int,
    ):
        super().__init__()
        
        # Embedding layer (no positional embeddings or tokenization)
        self.embedding = nn.Linear(input_dim, embed_dim, bias=False)
        init.xavier_uniform_(self.embedding.weight)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, 2)
        init.xavier_uniform_(self.linear.weight)
        self.regression_head = nn.Sequential(
            self.linear,
            nn.Sigmoid()
        )
    
    def forward(self, sequence, mask):
        input_embedding = self.embedding(sequence)
        output_embedding = self.encoder(input_embedding, src_key_padding_mask=mask.T)
        # The output embeddings for masked tokens are NaN
        output_embedding = output_embedding.nan_to_num_()
        # Calculate the average of the output embeddings along the sequence dimension
        # Averaging only not masked output embeddings
        eps = 1e-7
        avg_embedding = output_embedding.sum(axis=1) / ((~mask).sum(axis=1).unsqueeze(1) + eps)
        return self.regression_head(avg_embedding)
