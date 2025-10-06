import torch
import torch.nn as nn
import os
from ..kernels.kernel_wrappers import fused_bias_relu_inplace

class ConvWithFusedBiasReLU(nn.Module):
    """
    Wraps an existing Conv2d (bias=False) + bias Tensor, and applies conv -> fused(bias+relu)
    fused_bias_relu_inplace(tensor, bias) performs in-place bias add and ReLU.
    """
    def __init__(self, conv: nn.Conv2d, bias: torch.Tensor):
        super().__init__()
        # copy conv (so original model remains untouched)
        self.conv = conv
        # bias is a 1D Tensor of shape [out_channels]
        self.register_buffer('bias', bias)

    def forward(self, x):
        y = self.conv(x)
        # fused op modifies y in-place
        fused_bias_relu_inplace(y, self.bias)
        return y