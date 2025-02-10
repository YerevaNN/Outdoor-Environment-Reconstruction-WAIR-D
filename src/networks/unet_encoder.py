from pl_bolts.models.vision.unet import DoubleConv, Down
from torch import nn


class UNetEncoder(nn.Module):
    
    def __init__(self, input_channels: int, num_layers: int, features_start: int):
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")
        
        super().__init__()
        self.num_layers = num_layers
        
        layers = [DoubleConv(input_channels, features_start)]
        
        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2
        
        self.encoder = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(feats, 2),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        return self.encoder(image)
