"""Print a machine-readable RAIQ Colab/GPU environment report."""
from __future__ import annotations

import json
import platform
import shutil
import sys

import torch


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "free_disk_gib": round(shutil.disk_usage('/').free / 2**30, 2),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update({
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_vram_gib": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
