"""Lifecycle wrapper for the vLLM OpenAI-compatible server.

Usage:
    python -m src.server start                      # prefix caching ON (default)
    python -m src.server start --no-prefix-caching  # control run for Phase 3
    python -m src.server start --num-gpu-blocks 512 # starve the KV cache (PagedAttention lab)
    python -m src.server status
    python -m src.server stop

The server runs as a detached background process and logs to logs/vllm_server.log,
so it survives across Colab cells and can be restarted with different flags.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import requests

from src import config

LOG_FILE = config.LOG_DIR / "vllm_server.log"
PID_FILE = config.LOG_DIR / "vllm_server.pid"
STATE_FILE = config.LOG_DIR / "server_config.json"


def build_command(args) -> list[str]:
    gpu_name, cap = config.gpu_info()
    cmd = [
        "vllm", "serve", config.MODEL,
        "--host", config.HOST,
        "--port", str(config.PORT),
        "--max-model-len", str(config.MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--enable-prefix-caching" if args.prefix_caching else "--disable-prefix-caching",
    ]
    # Turing (T4, SM 7.5) has no bf16 and no FlashAttention-2 kernels:
    # force fp16 and the Triton attention backend. Ampere+/Ada (L4) use the defaults.
    if cap < (8, 0):
        cmd += ["--dtype", "half", "--attention-backend", "TRITON_ATTN"]
    if args.num_gpu_blocks:
        cmd += ["--num-gpu-blocks-override", str(args.num_gpu_blocks)]
    if args.max_num_seqs:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    cmd += args.extra
    return cmd


def is_healthy(timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{config.BASE_URL}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _pid():
    try:
        return int(PID_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _alive(pid) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(quiet: bool = False):
    pid = _pid()
    if _alive(pid):
        # vllm serve spawns engine-core worker processes; kill the whole group.
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        for _ in range(60):
            if not _alive(pid):
                break
            time.sleep(0.5)
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        if not quiet:
            print(f"Stopped vLLM server (pid {pid}).")
    elif not quiet:
        print("No running server found.")
    PID_FILE.unlink(missing_ok=True)
    # Give the driver a moment to release VRAM before a restart.
    time.sleep(3)


def start(args):
    if _alive(_pid()):
        print("A server is already running; restarting with new flags.")
    stop(quiet=True)

    config.LOG_DIR.mkdir(exist_ok=True)
    cmd = build_command(args)
    print("Launching:", " ".join(cmd))
    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    PID_FILE.write_text(str(proc.pid))

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"Server exited with code {proc.returncode}. Last log lines:")
            print(tail(40))
            sys.exit(1)
        if is_healthy():
            break
        time.sleep(3)
    else:
        print(f"Server not healthy after {args.timeout}s. Last log lines:")
        print(tail(40))
        sys.exit(1)

    info = {
        "model": config.MODEL,
        "gpu": config.gpu_info()[0],
        "prefix_caching": args.prefix_caching,
        "num_gpu_blocks_override": args.num_gpu_blocks,
        "max_model_len": config.MAX_MODEL_LEN,
        "command": cmd,
        **kv_cache_summary(),
    }
    STATE_FILE.write_text(json.dumps(info, indent=2))
    print(f"Server healthy at {config.API_URL}")
    for k in ("gpu_kv_cache_tokens", "max_concurrency_at_max_len"):
        if info.get(k):
            print(f"  {k}: {info[k]}")


def kv_cache_summary() -> dict:
    """Pull the KV-cache sizing lines vLLM prints at startup.

    These are the numbers PagedAttention is about: how many tokens of KV
    the pool holds, and how many *max-length* requests that would fit if
    memory were reserved contiguously per request.
    """
    text = LOG_FILE.read_text(errors="ignore") if LOG_FILE.exists() else ""
    out = {}
    m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if m:
        out["gpu_kv_cache_tokens"] = int(m[-1].replace(",", ""))
    m = re.findall(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", text)
    if m:
        out["max_concurrency_at_max_len"] = float(m[-1])
    return out


def tail(n: int = 30) -> str:
    if not LOG_FILE.exists():
        return "(no log file)"
    return "\n".join(LOG_FILE.read_text(errors="ignore").splitlines()[-n:])


def status():
    pid = _pid()
    print(f"pid={pid} alive={_alive(pid)} healthy={is_healthy()}")
    if STATE_FILE.exists():
        print(STATE_FILE.read_text())


def server_state() -> dict:
    """Config of the currently running server, recorded at start()."""
    try:
        return json.loads(STATE_FILE.read_text())
    except FileNotFoundError:
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("start")
    s.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    s.add_argument("--gpu-mem", type=float, default=0.90, help="--gpu-memory-utilization")
    s.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="Override KV-cache block count (forces KV memory pressure)")
    s.add_argument("--max-num-seqs", type=int, default=None)
    s.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (faster startup)")
    s.add_argument("--timeout", type=int, default=900)
    s.add_argument("extra", nargs=argparse.REMAINDER, help="Extra flags passed to vllm serve after --")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("logs")

    args = p.parse_args()
    if args.action == "start":
        args.extra = [a for a in args.extra if a != "--"]
        start(args)
    elif args.action == "stop":
        stop()
    elif args.action == "status":
        status()
    elif args.action == "logs":
        print(tail(60))


if __name__ == "__main__":
    main()
