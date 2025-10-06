#!/usr/bin/env python3
"""
Generate a single HTML dashboard from benchmark results.
"""

import os
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO

# ------------------------------------------------------------------------------
# Safe import for markdown and matplotlib
# ------------------------------------------------------------------------------
try:
    import markdown
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
SUMMARY_DIR = ROOT / "results_summary"
REPORT_HTML = SUMMARY_DIR / "report.html"

# Combined CSV & speedup plot
COMBINED_CSV = SUMMARY_DIR / "combined_results.csv"
SPEEDUP_PLOT = SUMMARY_DIR / "speedup_trend.png"
TOP_KERNELS_CSV = SUMMARY_DIR / "top_kernels.csv"

# ------------------------------------------------------------------------------
# Helper: Embed images as base64
# ------------------------------------------------------------------------------
def embed_image(path):
    if path.exists():
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.replace(".", "")
        return f'<img src="data:image/{ext};base64,{encoded}" style="max-width:800px;">'
    return ""

# ------------------------------------------------------------------------------
# Build Markdown content
# ------------------------------------------------------------------------------
md_content = "# NVIDIA Thunder Fusion Benchmark Dashboard\n\n"

# Summary table
if COMBINED_CSV.exists():
    df = pd.read_csv(COMBINED_CSV)
    md_content += "## Benchmark Summary\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n"
else:
    md_content += "## Benchmark Summary\nNo CSV summary found.\n\n"

# Speedup chart
md_content += "## Speedup Chart\n\n"
md_content += embed_image(SPEEDUP_PLOT) + "\n\n"

# Top kernels
if TOP_KERNELS_CSV.exists():
    topk_df = pd.read_csv(TOP_KERNELS_CSV)
    md_content += "## Top CUDA Kernels (Fused Model)\n\n"
    md_content += topk_df.to_markdown(index=False) + "\n\n"

# ------------------------------------------------------------------------------
# Convert Markdown to HTML
# ------------------------------------------------------------------------------
html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NVIDIA Thunder Fusion Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        img {{ display:block; margin: 20px 0; }}
        h1,h2,h3 {{ color: #2a7ae2; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""

# ------------------------------------------------------------------------------
# Save HTML dashboard
# ------------------------------------------------------------------------------
with open(REPORT_HTML, "w") as f:
    f.write(html_page)

print(f"[OK] HTML dashboard saved → {REPORT_HTML}")
