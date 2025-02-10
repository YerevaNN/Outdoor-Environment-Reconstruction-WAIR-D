import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import xavier_init
from torch import nn


class StepNeck(nn.Module):
    
    def __init__(self, in_channels: list[int], out_channels: list[int], scales: list[float]):
        super().__init__()
        assert isinstance(in_channels, list)
        assert len(in_channels) == len(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                )
            )
            self.convs.append(
                ConvModule(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    
    def forward(self, inputs):
        return [
            self.convs[i](
                F.interpolate(
                    self.lateral_convs[i](inputs[i]),
                    scale_factor=self.scales[i], mode='bilinear', align_corners=True
                )
            )
            for i in range(self.num_outs)
        ]
