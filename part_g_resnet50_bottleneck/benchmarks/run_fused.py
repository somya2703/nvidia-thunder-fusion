#!/usr/bin/env python3
"""
Run the baseline ResNet50 benchmark and save results.
"""

import os
import json
import time
import torch
from part_g_resnet50_bottleneck.models.resnet50_baseline import get_resnet50_baseline
from part_g_resnet50_bottleneck.benchmarks.benchmark_utils import run_and_report

RESULTS_DIR = "benchmarks/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.json")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fn = lambda x: get_resnet50_baseline()(x.to(device))

    # Run benchmark
    y = run_and_report(model_fn, name="Baseline ResNet50", device=device)

    # Example: measure latency and throughput
    torch.cuda.synchronize()
    start = time.time()
    y = model_fn(torch.randn(8, 3, 224, 224, device=device))
    torch.cuda.synchronize()
    end = time.time()
    latency_ms = (end - start) * 1000
    throughput_img_s = 8 / (end - start)

    # Load previous results if exist, to merge
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Update baseline results
    results.update({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "baseline_latency_ms": latency_ms,
        "baseline_throughput_img_s": throughput_img_s,
        # Compute speedup if fused data exists
        "speedup": None
    })

    # Compute speedup if fused exists
    if "fused_latency_ms" in results and results["fused_latency_ms"]:
        results["speedup"] = results["baseline_latency_ms"] / results["fused_latency_ms"]

    # Write updated results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f" Baseline model benchmark complete")
    print(f"Latency per batch: {latency_ms:.2f} ms")
    print(f"Throughput: {throughput_img_s:.2f} images/s")
    print(f"Results written to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
