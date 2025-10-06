#!/usr/bin/env python3
"""
Analyze and visualize benchmark and profiling results for NVIDIA Thunder Fusion.
Generates a combined summary of all runs + speedup charts.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Safe import for seaborn
# ------------------------------------------------------------------------------
try:
    import seaborn as sns
except ImportError:
    print("[INFO] seaborn not found — installing it automatically...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

# ------------------------------------------------------------------------------
# 2. Paths and setup
# ------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
SUMMARY_DIR = os.path.join(ROOT, "results_summary")
BENCH_DIR = os.path.join(SUMMARY_DIR)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"[INFO] Loading results from: {BENCH_DIR}")

# ------------------------------------------------------------------------------
# 3. Collect all benchmark result JSON files
# ------------------------------------------------------------------------------
records = []
for f in os.listdir(BENCH_DIR):
    if f.startswith("benchmark_results_") and f.endswith(".json"):
        fpath = os.path.join(BENCH_DIR, f)
        try:
            with open(fpath, "r") as fh:
                data = json.load(fh)
                records.append(data)
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

if not records:
    print("[ERROR] No benchmark results found in results_summary/. Run ./run_all.sh first.")
    exit(1)

# ------------------------------------------------------------------------------
# 4. Convert to DataFrame and clean
# ------------------------------------------------------------------------------
df = pd.DataFrame(records)
expected_cols = ["timestamp", "gpu_name", "baseline_latency_ms", "fused_latency_ms", "speedup"]

# Add missing columns safely
for c in expected_cols:
    if c not in df.columns:
        df[c] = None

# Sort chronologically if timestamp exists
if "timestamp" in df.columns and df["timestamp"].notnull().any():
    df = df.sort_values("timestamp")
else:
    df["timestamp"] = [f"run_{i}" for i in range(len(df))]

print(f"[OK] Loaded {len(df)} runs.")
print(df[["timestamp", "gpu_name", "baseline_latency_ms", "fused_latency_ms", "speedup"]].tail())

# ------------------------------------------------------------------------------
# 5. Save combined summary CSV
# ------------------------------------------------------------------------------
summary_csv = os.path.join(SUMMARY_DIR, "combined_results.csv")
df.to_csv(summary_csv, index=False)
print(f"[OK] Combined results saved → {summary_csv}")

# ------------------------------------------------------------------------------
# 6. Generate Speedup Plot
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="timestamp",
    y="speedup",
    hue="gpu_name",
    palette="viridis"
)
plt.title("Fusion Speedup Over Time")
plt.ylabel("Speedup (×)")
plt.xlabel("Run Timestamp")
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = os.path.join(SUMMARY_DIR, "speedup_trend.png")
plt.savefig(plot_path)
print(f"[OK] Speedup chart saved → {plot_path}")

# ------------------------------------------------------------------------------
# 7. Profiling Summary (optional)
# ------------------------------------------------------------------------------
profile_dir = os.path.join(ROOT, "part_g_resnet50_bottleneck", "benchmarks", "results", "profiling")
kernel_csv = os.path.join(profile_dir, "fused_kernels.csv")
if os.path.exists(kernel_csv):
    prof_df = pd.read_csv(kernel_csv)
    topk = prof_df.nlargest(10, "cuda_time_total_ms")
    prof_summary_csv = os.path.join(SUMMARY_DIR, "top_kernels.csv")
    topk.to_csv(prof_summary_csv, index=False)
    print(f"[OK] Profiling kernel summary saved → {prof_summary_csv}")
else:
    print("[INFO] No profiling data found — skipping kernel summary.")

print("\n All analysis complete.")
