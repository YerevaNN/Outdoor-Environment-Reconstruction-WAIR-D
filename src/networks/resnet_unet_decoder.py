import torch.nn as nn
from pl_bolts.models.vision.unet import DoubleConv, Up
from torchvision.models import resnet50, ResNet50_Weights


class ResNetUNetDecoder(nn.Module):
    
    def __init__(self, num_classes: int, input_channels: int):
        super().__init__()
        self.pre_conv = DoubleConv(input_channels, 3)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        u_layers = []
        features = [
            self.resnet.layer4[2].bn3.num_features,
            self.resnet.layer3[5].bn3.num_features,
            self.resnet.layer2[3].bn3.num_features,
            self.resnet.layer1[2].bn3.num_features,
            self.resnet.bn1.num_features,
        ]
        for feats_in, feats_out in zip(features[:-2], features[1:-1]):
            u_layers.append(Up(feats_in, feats_out, False))
        u_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(feats_out, features[-1], kernel_size=2, stride=2),
                DoubleConv(features[-1], features[-1]),
                nn.ConvTranspose2d(features[-1], features[-1], kernel_size=2, stride=2),
                DoubleConv(features[-1], features[-1])
            )
        )
        u_layers.append(nn.Conv2d(features[-1], num_classes, kernel_size=1))
        self.u_layers = nn.ModuleList(u_layers)
    
    def forward(self, image):
        x = self.pre_conv(image)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        xi = [x]
        # Down path
        for layer in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            xi.append(layer(xi[-1]))
        
        # Up path
        for i, layer in enumerate(self.u_layers[: -2]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        
        return self.u_layers[-1](self.u_layers[-2](xi[-1]))
