import subprocess
import sys
from pathlib import Path
import shutil
import os
root = Path(__file__).parent.parent  # points to part_g_resnet50_bottleneck
setup_path = root / "setup.py"

def clean_build():
    # Remove build directory
    build_dir = root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    # Remove all compiled .so files
    for so_file in root.glob("**/*.so"):
        so_file.unlink()

def build_extension():
    subprocess.run(
        [sys.executable, str(setup_path), "build_ext", "--inplace"],
        cwd=root,
        check=True
    )

if __name__ == "__main__":
    clean_build()
    build_extension()
