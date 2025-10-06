import torch
import os
from part_g_resnet50_bottleneck.models.resnet50_fused import get_fused_resnet50

def test_forward_pass():
    model = get_fused_resnet50(device="cuda")
    x = torch.randn(2, 3, 224, 224, device="cuda")
    y = model(x)
    assert y.shape[-1] == 1000
    print("Integration test passed ✅")
