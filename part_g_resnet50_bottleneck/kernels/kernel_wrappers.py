import os
import torch
from torch.utils.cpp_extension import load

root = os.path.dirname(__file__)
_fused = load(
    name="fused_kernels",
    sources=[os.path.join(root, "conv_bn_relu_kernel.cu")],
    verbose=True,
)

def conv_bn_relu(x, bias):
    """Run fused Conv+Bias+ReLU kernel"""
    y = torch.empty_like(x)
    _fused.conv_bn_relu_launcher(x, bias, y)
    return y
