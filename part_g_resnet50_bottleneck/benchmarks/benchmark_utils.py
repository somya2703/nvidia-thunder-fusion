import time
import torch
import os
def run_and_report(model_fn, name="model", device="cuda", input_shape=(8, 3, 224, 224)):
    x = torch.randn(*input_shape, device=device)
    torch.cuda.synchronize()
    start = time.time()
    y = model_fn(x)
    torch.cuda.synchronize()
    end = time.time()
    print(f"{name} runtime: {end - start:.5f}s, output norm: {y.norm().item():.5f}")
    return y

def assert_allclose(a, b, tol=1e-3):
    if not torch.allclose(a, b, atol=tol, rtol=tol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"Outputs differ (max diff {diff:.6f})")
    print("✅ Outputs match within tolerance.")
