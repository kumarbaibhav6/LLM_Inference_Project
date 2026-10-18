# PROJECT CONTEXT & PLAN: High-Throughput Custom Model Serving Infrastructure

## 1. Purpose
The objective of this project is to build and benchmark a custom, high-throughput LLM serving microservice using `vLLM`. This moves beyond basic API-wrapping to demonstrate production-grade MLOps infrastructure skills. The project will run locally (Mac M2) or via Google Colab (T4/L4 GPU) using an optimized 1.5B parameter model to simulate enterprise-scale concurrency.

## 2. Learning Goals & Architectural Focus
*   **KV Cache & Memory Bottlenecks:** Demonstrate the $O(N^2)$ prefill compute burden versus the $O(N)$ decode memory bottleneck.
*   **PagedAttention:** Validate how block chunking eliminates VRAM fragmentation.
*   **Automatic Prefix Caching:** Measure the drastic reduction in Time-To-First-Token (TTFT) when using Radix tree cache detection for heavy, static system prompts.
*   **Continuous Batching:** Prove the engine can keep the GPU saturated under asynchronous, concurrent load without traditional static batching idle time.

---

## 3. Execution Phases & Action Items

*Agent Instructions: Execute this plan one phase at a time. After completing a phase, run the verification steps. Do not proceed to the next phase until verification passes. Update the checkboxes `[x]` in this file as you complete each step.*

### Phase 1: Environment & Project Initialization
- [x] Create the project directory structure: `src/`, `benchmarks/`, and `docs/`.
- [x] Initialize the Python environment and write a `requirements.txt` including: `vllm`, `openai`, `pandas`, `matplotlib`, `requests`, `nest_asyncio`.
- [ ] **Verification:** Ensure dependencies can be installed without version conflicts. _(pending: run on Colab GPU)_

### Phase 2: Server Deployment Script
- [x] Create a shell script (`start_server.sh`) or Python wrapper to launch the `vLLM` OpenAI-compatible API server in the background.
- [x] Configure the server with the following parameters:
    - Model: `Qwen/Qwen2.5-1.5B-Instruct`
    - Flags: `--enable-prefix-caching`, `--max-model-len 4096`, `--port 8000`
- [ ] **Verification:** The server successfully exposes `http://localhost:8000/v1/chat/completions` and logs indicate the model is loaded into VRAM. _(pending: run on Colab GPU)_

### Phase 3: Prefix Caching Evaluation (`src/test_cache.py`)
- [x] Write a synchronous Python script using the `openai` client.
- [x] Define a heavy, static system prompt (>500 words) to simulate an enterprise RAG context.
- [x] **Query 1 (Cold Start):** Send the prompt + question. Calculate and log the TTFT and total latency.
- [x] **Query 2 (Cache Hit):** Send the *exact same* system prompt + a *different* question. Calculate and log the TTFT and total latency.
- [x] Calculate and print the percentage latency reduction.
- [ ] **Verification:** Running the script must output a massive latency drop for Query 2 compared to Query 1. _(pending: run on Colab GPU)_

### Phase 4: Concurrency Load Test (`src/load_test.py`)
- [x] Write an asynchronous script using `asyncio` and `AsyncOpenAI`. (Include `nest_asyncio.apply()` to support Jupyter/Colab environments).
- [x] Write a function to fire 25 concurrent requests simultaneously to the server.
- [x] Log the individual completion times and the total time taken for the batch.
- [ ] **Verification:** All 25 requests complete successfully without server timeout or Out-Of-Memory (OOM) errors, proving continuous batching is active. _(pending: run on Colab GPU)_

### Phase 5: Data Visualization & Packaging
- [x] Create `src/plot_metrics.py` to generate Matplotlib bar charts comparing the TTFT of the Cold Start vs. Cache Hit.
- [x] Save the generated charts to the `/benchmarks` folder.
- [x] Write a comprehensive `README.md` explaining the architecture, the impact of PagedAttention, and the Prefix Caching metrics.
- [x] Create a `Dockerfile` for containerized deployment of the vLLM server.
- [ ] **Verification:** Code is clean, modularized, and the repository is ready for a portfolio review. _(pending: run on Colab GPU)_

---

## 4. Implementation Notes & Deviations

Code for every phase is written and smoke-tested locally against a mock OpenAI-compatible server
(no GPU on the Mac). The **Verification** steps need a GPU: run `notebooks/colab_runner.ipynb`, then tick them.

* **Mac M2 dropped as a target.** vLLM's GPU path is CUDA-only, so all GPU work runs on Colab (L4 recommended, T4 supported via `--dtype half --attention-backend TRITON_ATTN`).
* **vLLM pinned to 0.30.0.** Flags and metric names drift between minor releases (e.g. `--disable-prefix-caching`, `vllm:kv_cache_usage_perc`).
* **"Radix tree" corrected.** That is SGLang's RadixAttention; vLLM's APC uses chained block hashes. See `docs/concepts.md` section 4.
* **Phase 3 hardened:** streaming TTFT, warm-up, 5 trials × 4 prompt lengths (750-3500 tokens), a nonce to guarantee cold starts, Prometheus cache-hit counts, and a caching-disabled control run.
* **Phase 4 extended:** besides the 25-request burst, static vs. continuous batching on identical requests, a concurrency sweep, and server-side metrics sampling. A burst finishing without OOM alone doesn't prove continuous batching.
* **Added Phase 0 (`src/kv_cache_demo.py`):** KV cache on/off, prefill vs. decode, batch-decode scaling, and contiguous vs. paged allocation math. This makes the "KV cache & memory bottlenecks" goal measurable.
* **Added PagedAttention lab:** a KV-starved server (`--num-gpu-blocks 1024`) under 128 concurrent requests, showing admission above the contiguous limit plus preemption. The original plan had no PagedAttention experiment.
* **Added `docs/concepts.md`** (theory + interview questions) and `notebooks/colab_runner.ipynb` (end-to-end driver).

