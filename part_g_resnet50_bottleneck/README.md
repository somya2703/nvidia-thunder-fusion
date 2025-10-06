# NVIDIA Thunder Fusion

High-performance fused ResNet50 benchmarking and analysis project leveraging PyTorch, CUDA, and NVIDIA GPU acceleration.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Resetting the Project](#resetting-the-project)
- [Analyzing Results](#analyzing-results)
- [Generating HTML Dashboard](#generating-html-dashboard)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Overview
NVIDIA Thunder Fusion demonstrates optimized ResNet50 performance with CUDA fused kernels. It provides:

- Baseline and fused ResNet50 benchmarking
- PyTorch profiling
- Aggregated metrics (latency, throughput, GPU utilization)
- Visual plots (speedup, GPU memory usage)
- HTML dashboard for easy reporting

---

## Project Structure
```
nvidia-thunder-fusion/
├── part_g_resnet50_bottleneck/
│   ├── benchmarks/
│   │   ├── results/
│   │   └── run_baseline.py
│   │   └── run_fused.py
│   ├── kernels/
│   │   └── build.py
│   ├── models/
│   │   └── resnet50_fused.py
│   └── tests/
├── results_summary/
├── scripts/
│   ├── analyze_results_pro.py
│   ├── generate_dashboard.py
│   └── reset_project.py
├── run_all.sh
└── README.md
```

---

## Setup
1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- torch
- torchvision
- pandas
- numpy
- matplotlib
- seaborn
- lark-parser
- tabulate
- markdown

---

## Running the Full Pipeline
Execute the all-in-one script:
```bash
./run_all.sh
```
This will:
1. Activate the virtual environment
2. Check system GPU and CUDA
3. Clean previous CUDA builds and compile fused kernels
4. Run baseline and fused benchmarks
5. Aggregate benchmark results into JSON
6. Run integration/unit tests
7. Run PyTorch profiler
8. Analyze results (speedup plots, GPU metrics)
9. Generate HTML dashboard at `results_summary/report.html`

---

## Resetting the Project
To start a fresh experiment, remove all previous results, builds, and caches:
```bash
python scripts/reset_project.py
```

This will clean:
- `results_summary/`
- `part_g_resnet50_bottleneck/build/`
- Old benchmark JSON, CSV, PNG, log files

---

## Analyzing Results
Run the analysis script independently if needed:
```bash
python scripts/analyze_results_pro.py
```
It will:
- Load all benchmark JSON results
- Generate speedup plots using Seaborn/Matplotlib
- Print tabular summaries

---

## Generating HTML Dashboard
After analysis, generate a consolidated HTML report:
```bash
python scripts/generate_dashboard.py
```
- Converts Markdown summary to HTML
- Embeds plots and tables
- Output: `results_summary/report.html`

---

## Results
Current outputs include:
- **JSON:** timestamped benchmark results (`results_summary/benchmark_results_*.json`)
- **Plots:** speedup, GPU memory usage (`results_summary/*.png`)
- **Logs:** build and test logs (`results_summary/*.log`)
- **Profiler traces:** PyTorch fused model traces (`part_g_resnet50_bottleneck/benchmarks/results/profiling/fused_trace.json`)
- **HTML Dashboard:** consolidated `report.html`

Optional improvements:
- Per-layer kernel timings
- GPU utilization and memory consumption plots
- Comparative results for multiple GPUs or configurations
- Automated HTML dashboard generation with embedded plots and tables

---

## Dependencies
Make sure the following Python packages are installed in your `.venv`:
```
torch
torchvision
pandas
numpy
matplotlib
seaborn
lark-parser
tabulate
markdown
```

CUDA toolkit 12.8+ and a compatible NVIDIA GPU are required for fused kernel execution and profiling.

