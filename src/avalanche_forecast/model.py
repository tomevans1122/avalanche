"""ResNet-50 image encoder plus temporal fusion for binary event likelihood."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class TemporalResNet50(nn.Module):
    """Encode T satellite frames and aligned measurements into one logit.

    Inputs are image tensors `[batch, T, channels, height, width]` and weather /
    snowpack tensors `[batch, T, meteo_features]`.  Logits are deliberately
    returned instead of thresholded labels, allowing calibration downstream.
    """

    def __init__(self, image_channels: int, meteo_features: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            image_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        with torch.no_grad():
            # Transfer RGB filters to arbitrary sensor bands without discarding them.
            source = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.conv1.weight.copy_(source.repeat(1, image_channels, 1, 1))
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.temporal = nn.GRU(2048 + meteo_features, 256, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.25), nn.Linear(256, 1))

    def forward(self, imagery: torch.Tensor, meteo: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = imagery.shape
        if meteo.shape[:2] != (batch, steps):
            raise ValueError("meteo must share batch and time dimensions with imagery")
        encoded = self.image_encoder(imagery.reshape(batch * steps, channels, height, width))
        encoded = encoded.flatten(1).reshape(batch, steps, -1)
        history, _ = self.temporal(torch.cat([encoded, meteo], dim=-1))
        return self.head(history[:, -1]).squeeze(-1)
