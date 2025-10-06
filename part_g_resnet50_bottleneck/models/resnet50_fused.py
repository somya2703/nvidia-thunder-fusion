import torch
import torch.nn as nn
import os
from part_g_resnet50_bottleneck.kernels import kernel_wrappers as kw


class FusedResNet50Block(nn.Module):
    """A simplified fused bottleneck-like block (Conv + BN + ReLU)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Run conv + batchnorm
        y = self.conv(x)
        y = self.bn(y)
        # Apply custom fused CUDA kernel if available
        if hasattr(kw, "conv_bn_relu"):
            bias = self.conv.bias if self.conv.bias is not None else torch.zeros_like(y[0, :, 0, 0])
            y = kw.conv_bn_relu(y, bias)
        else:
            y = torch.relu(y)
        return y


def get_fused_resnet50(device="cpu"):
    """Return a small ResNet50-like model using fused blocks."""
    model = nn.Sequential(
        FusedResNet50Block(3, 64, kernel_size=7, stride=2, padding=3),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        FusedResNet50Block(64, 128),
        FusedResNet50Block(128, 256),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 1000),
    )
    return model.to(device)
