# vLLM OpenAI-compatible server for Qwen2.5-1.5B-Instruct.
#   docker build -t llm-serving .
#   docker run --gpus all -p 8000:8000 --ipc=host \
#       -v ~/.cache/huggingface:/root/.cache/huggingface llm-serving
# Same vLLM version as requirements.txt so benchmarks and container agree.
ARG VLLM_VERSION=v0.30.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

# Shell form so env vars expand; `exec` keeps vLLM as PID 1 for clean SIGTERM handling.
# Extra flags can be appended: docker run ... llm-serving --max-num-seqs 128
ENTRYPOINT ["/bin/bash", "-c", "exec vllm serve \"$MODEL\" --host 0.0.0.0 --port 8000 --max-model-len \"$MAX_MODEL_LEN\" --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" --enable-prefix-caching \"$@\"", "--"]
