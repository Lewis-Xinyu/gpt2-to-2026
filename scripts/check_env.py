"""
Print the runtime facts that matter before launching a training run.

Run this on the Windows RTX 5070 machine after installing dependencies:
  python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda runtime: {torch.version.cuda}")
        print(f"cuda devices: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / 1024**3
            print(f"cuda:{index}: {props.name} ({total_gb:.1f} GB)")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"mps available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
