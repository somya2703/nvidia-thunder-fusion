#!/usr/bin/env python3
"""
Thunder Fusion Project Reset Utility
------------------------------------
Safely cleans up all generated results, build artifacts, caches, and temporary files.

Run this when you want to start a fresh experiment:
    python scripts/reset_project.py
"""

import os
import shutil
from pathlib import Path
import glob

# -----------------------------
# Config: directories and files
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent

# Directories to remove
TARGET_DIRS = [
    ROOT / "results_summary",                         # aggregated metrics and reports
    ROOT / "part_g_resnet50_bottleneck" / "build",    # CUDA build artifacts
    ROOT / "part_g_resnet50_bottleneck" / "__pycache__",
    ROOT / ".cache",                                  # Torch cache (optional)
]

# File patterns to remove (relative to ROOT)
TARGET_FILES = [
    ROOT / "part_g_resnet50_bottleneck" / "benchmarks" / "results" / "*",  # old benchmark outputs
    ROOT / "results_summary" / "*.json",
    ROOT / "results_summary" / "*.png",
    ROOT / "results_summary" / "*.csv",
    ROOT / "results_summary" / "*.log",
    ROOT / "results_summary" / "report_*.md",
]

# -----------------------------
# Helper function
# -----------------------------

def delete_path(path: Path):
    """Delete a file or directory safely."""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"  Deleted directory: {path}")
        else:
            path.unlink()
            print(f"  Deleted file: {path}")
    except Exception as e:
        print(f"[WARN] Could not delete {path}: {e}")

# -----------------------------
# Main cleanup
# -----------------------------

def clean_project():
    print("-------------------------------------------------------")
    print(" Thunder Fusion Project Reset — Clean Slate Mode")
    print("-------------------------------------------------------\n")

    # Remove directories
    for d in TARGET_DIRS:
        delete_path(d)

    # Remove files by glob pattern
    for pattern in TARGET_FILES:
        for path in glob.glob(str(pattern)):
            delete_path(Path(path))

    print("\n Cleanup complete! Your project is now fresh and ready.\n")
    print("Next steps:")
    print("    Rebuild CUDA extensions and run benchmarks: ./run_all.sh")
    print("    Generate dashboard: ./run_all.sh will automatically generate report.html")

# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    clean_project()
