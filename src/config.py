"""Shared settings for the server launcher and all benchmark scripts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
LOG_DIR = ROOT / "logs"

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))

BASE_URL = f"http://{HOST}:{PORT}"
API_URL = f"{BASE_URL}/v1"
API_KEY = "EMPTY"  # vLLM ignores it unless started with --api-key


def gpu_info():
    """Return (name, compute_capability) of GPU 0 without importing torch.

    The notebook kernel never imports torch, so the vLLM server process
    (launched separately) is the only thing holding GPU memory.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        name, cap = [s.strip() for s in out.split(",")]
        major, minor = cap.split(".")
        return name, (int(major), int(minor))
    except Exception:
        return "unknown", (0, 0)


def gpu_tag() -> str:
    """Short label like 'L4' or 'T4' used in result filenames."""
    name, _ = gpu_info()
    for tag in ("L4", "T4", "A100", "H100", "A10G"):
        if tag in name:
            return tag
    return name.replace(" ", "_") or "gpu"
