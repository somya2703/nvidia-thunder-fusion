#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# NVIDIA Thunder Fusion - Complete Build, Test & Benchmark Pipeline
# ------------------------------------------------------------------------------

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARKS_DIR="$PROJECT_ROOT/part_g_resnet50_bottleneck/benchmarks"
RESULTS_DIR="$BENCHMARKS_DIR/results"
SUMMARY_DIR="$PROJECT_ROOT/results_summary"
PROFILING_DIR="$RESULTS_DIR/profiling"

mkdir -p "$RESULTS_DIR" "$PROFILING_DIR" "$SUMMARY_DIR"

echo "=== NVIDIA Thunder Fusion: Run All ==="
echo "Project root: $PROJECT_ROOT"
echo "Benchmark results: $RESULTS_DIR"
echo "Summary outputs: $SUMMARY_DIR"
echo "-------------------------------------"

# ------------------------------------------------------------------------------
# 1. Activate environment
# ------------------------------------------------------------------------------

if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "[ERROR] Virtual environment not found!"
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Environment checks
# ------------------------------------------------------------------------------

echo "[INFO] Checking system..."

python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY

nvidia-smi || echo "[WARN] nvidia-smi not found"
nvcc --version || echo "[WARN] nvcc not found"

# ------------------------------------------------------------------------------
# 3. Clean build CUDA kernel
# ------------------------------------------------------------------------------

echo "[BUILD] Cleaning previous build..."
python "$PROJECT_ROOT/part_g_resnet50_bottleneck/kernels/build.py" clean || echo "[WARN] Clean failed"

echo "[BUILD] Compiling CUDA fused kernels..."
python "$PROJECT_ROOT/part_g_resnet50_bottleneck/kernels/build.py" > "$SUMMARY_DIR/build_log.txt" 2>&1
echo "[OK] Build finished. Logs: $SUMMARY_DIR/build_log.txt"

# ------------------------------------------------------------------------------
# 4. Run Baseline & Fused Benchmarks
# ------------------------------------------------------------------------------

echo "[RUN] Baseline benchmark..."
python -m part_g_resnet50_bottleneck.benchmarks.run_baseline | tee "$RESULTS_DIR/baseline_stdout.txt"

echo "[RUN] Fused benchmark..."
python -m part_g_resnet50_bottleneck.benchmarks.run_fused | tee "$RESULTS_DIR/fused_stdout.txt"

# ------------------------------------------------------------------------------
# 5. Generate Detailed JSON Summary (timestamped + latest)
# ------------------------------------------------------------------------------

echo "[INFO] Collecting and writing benchmark results..."
python - <<'PY'
import re, json, os, torch, datetime

BENCHMARKS_DIR = "part_g_resnet50_bottleneck/benchmarks"
RESULTS_DIR = os.path.join(BENCHMARKS_DIR, "results")
SUMMARY_DIR = "results_summary"
os.makedirs(SUMMARY_DIR, exist_ok=True)

def extract_metrics(path):
    with open(path) as f:
        text = f.read()
    latency = None
    throughput = None
    if m := re.findall(r"([0-9.]+)\s*ms", text):
        latency = sum(map(float, m)) / len(m)
    if t := re.search(r"Throughput:\s*([0-9.]+)\s*images/s", text):
        throughput = float(t.group(1))
    return latency, throughput

baseline_lat, baseline_tp = extract_metrics(os.path.join(RESULTS_DIR, "baseline_stdout.txt"))
fused_lat, fused_tp = extract_metrics(os.path.join(RESULTS_DIR, "fused_stdout.txt"))

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
speedup = round(baseline_lat / fused_lat, 3) if baseline_lat and fused_lat else None

results = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "gpu_name": gpu_name,
    "gpu_memory_GB": round(gpu_mem, 2),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "baseline_latency_ms": baseline_lat,
    "fused_latency_ms": fused_lat,
    "baseline_throughput_img_s": baseline_tp,
    "fused_throughput_img_s": fused_tp,
    "speedup": speedup,
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = os.path.join(SUMMARY_DIR, f"benchmark_results_{timestamp}.json")
latest_path = os.path.join(SUMMARY_DIR, "benchmark_results.json")

with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
with open(latest_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Results written to {json_path}")
PY

# ------------------------------------------------------------------------------
# 6. Run Tests
# ------------------------------------------------------------------------------

echo "[TEST] Running integration/unit tests..."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q part_g_resnet50_bottleneck/tests | tee "$SUMMARY_DIR/test_log.txt" || echo "[WARN] Some tests failed"

# ------------------------------------------------------------------------------
# 7. PyTorch Profiler (Fused model)
# ------------------------------------------------------------------------------

echo "[PROFILE] Running PyTorch profiler (fused model)..."

python - <<'PY'
import torch, torch.profiler, os
from part_g_resnet50_bottleneck.models.resnet50_fused import get_fused_resnet50

PROFILING_DIR = "part_g_resnet50_bottleneck/benchmarks/results/profiling"
os.makedirs(PROFILING_DIR, exist_ok=True)

model = get_fused_resnet50().cuda()
x = torch.randn(4, 3, 224, 224, device="cuda")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(x)

trace_path = os.path.join(PROFILING_DIR, "fused_trace.json")
prof.export_chrome_trace(trace_path)
print(f"[OK] Trace saved to {trace_path}")
PY

# ------------------------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------------------------

echo "-------------------------------------"
echo "[SUMMARY]"

latest_json=$(ls -t "$SUMMARY_DIR"/benchmark_results_*.json 2>/dev/null | head -n 1)

if [ -n "$latest_json" ]; then
    cat "$latest_json"
    echo "-------------------------------------"
    echo "✅ All done! Results saved to:"
    echo "  $RESULTS_DIR"
    echo "Aggregated summary JSON:"
    echo "  $latest_json"
else
    echo "[WARN] No summary JSON file found!"
fi

# ------------------------------------------------------------------------------
# 9. Analyze results and generate plots
# ------------------------------------------------------------------------------

echo "[ANALYZE] Processing benchmark results and generating plots..."
python scripts/analyze_results_pro.py

# ------------------------------------------------------------------------------
# 10. Generate HTML Dashboard
# ------------------------------------------------------------------------------

echo "[REPORT] Generating HTML dashboard..."
python scripts/generate_dashboard.py
