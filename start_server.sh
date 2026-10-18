#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server in the background and wait for /health.
#   ./start_server.sh                       # prefix caching on
#   ./start_server.sh --no-prefix-caching   # control run
#   ./start_server.sh --num-gpu-blocks 512  # KV-cache pressure experiment
# Stop with: python -m src.server stop
set -euo pipefail
cd "$(dirname "$0")"
exec python -m src.server start "$@"
