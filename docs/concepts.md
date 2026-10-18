# LLM inference concepts: what each experiment proves

Notes to make sure you can *explain* every number this repo produces. Each section ends
with the experiment that demonstrates it and the questions an interviewer is likely to ask.

---

## 1. The two phases: prefill and decode

Autoregressive generation has two phases with opposite hardware profiles.

**Prefill** processes the whole prompt in one forward pass. All N tokens go through the
weight matrices together (a matrix-matrix multiply), so each weight byte loaded from HBM is
reused N times. This phase is **compute-bound**. Attention scores are an N×N matrix, so the
attention part is O(N²). For a 1.5B model at a few thousand tokens, though, the linear layers
(O(N·d²)) still dominate, and prefill time grows roughly linearly with only mild curvature.
**Prefill time sets TTFT.**

**Decode** produces one token per step. Each step multiplies a single vector by every weight
matrix (matrix-vector), so all ~3 GB of fp16 weights *plus* the whole KV cache are read from
HBM to do ~2 FLOPs per parameter. Arithmetic intensity is about 1 FLOP/byte, while the
GPU's balance point is roughly 200 FLOPs/byte (L4: ~121 TFLOPs fp16 vs. ~300 GB/s). This
phase is **memory-bandwidth-bound**. **Decode speed sets TPOT** (time per output token).

Back-of-envelope decode floor at batch 1: 3.1 GB weights / 300 GB/s ≈ **10 ms/token**
(~100 tok/s). On a 1.5B model, kernel-launch overhead pushes the real number above that floor.
Note that T4 (~320 GB/s) and L4 (~300 GB/s) have *similar* bandwidth, so their single-stream
decode speed is close even though the L4 has far more compute. The L4 pulls ahead on prefill
and at high batch.

> Experiment: `kv_cache_demo.py` → `prefill_vs_decode_<gpu>.png`. Prefill time rises steeply
> with context while a decode step stays nearly flat.

**Likely questions:** *Why is TTFT dominated by prompt length but TPOT isn't?* *Is decode
compute- or memory-bound, and how would you check?* (Roofline: arithmetic intensity vs.
the GPU's FLOPs/bandwidth ratio.)

---

## 2. KV cache

Self-attention at step t needs keys and values for tokens 0..t-1. Those never change, so
they are computed once and stored. Without a cache, every step reruns the whole sequence:
O(N) work per token and O(N²) for the whole generation.

Size per token = `2 (K,V) × layers × kv_heads × head_dim × bytes`.
For Qwen2.5-1.5B: `2 × 28 × 2 × 128 × 2 B = 28 KiB/token`, so one 4096-token sequence takes
112 MiB. The model uses **GQA** (12 query heads share 2 KV heads), which makes its KV cache 6×
smaller than full multi-head attention. GQA is one of the main reasons modern models are cheap to serve.

At serving scale **the KV cache, not the weights, limits concurrency**. Once the weights are
loaded, every remaining GB is KV capacity, and the number of concurrent sequences is
`KV_budget / per-sequence KV`.

> Experiment: `kv_cache_demo.py` → `kv_cache_vs_none_<gpu>.png` (no-cache step time climbs
> linearly; cached step time stays flat) and the KV math printout.

**Likely questions:** *Compute the KV size for model X.* *How do GQA/MQA, KV quantization
(fp8 KV) or sliding-window attention change capacity?*

---

## 3. PagedAttention

**The problem:** before vLLM, servers reserved one contiguous KV buffer per request sized
for the *maximum* possible length. Real requests are much shorter, so most reserved memory
sat empty (internal fragmentation), and variable-sized buffers also fragmented the free space
(external fragmentation). The vLLM paper measured only 20-40% of KV memory actually holding tokens.

**The fix:** borrow virtual memory from operating systems. The KV cache is split into fixed-size **blocks**
(16 tokens by default). Each sequence has a **block table** that maps logical blocks to
physical blocks anywhere in a preallocated pool. Blocks are allocated *on demand* as the
sequence grows, so waste is less than one block per sequence. The attention kernel is rewritten to
gather K/V through the block table.

Consequences:
* Near-100% KV utilization means several times more concurrent sequences means higher throughput.
* Blocks can be **shared** (reference-counted) across sequences. That is the mechanism behind
  prefix caching and parallel sampling / beam search.
* When the pool is exhausted, the scheduler **preempts** a sequence (frees its blocks and later
  recomputes them) instead of crashing with OOM.

What to watch: at startup vLLM logs the KV pool size in tokens and "maximum concurrency for 4096
tokens per request". The second number is the contiguous-allocation bound. Under real traffic,
`vllm:num_requests_running` goes well above it.

> Experiments: `kv_cache_demo.py` → `kv_fragmentation.png` (simulation: contiguous ≈ 20%
> utilization, paged ≈ 99%). PagedAttention lab (`--num-gpu-blocks 1024`, `load_test --scenario
> pressure`) → `kv_pressure_<gpu>.png`: running requests above the contiguous limit, and
> preemptions instead of OOM when the pool runs dry.

**Likely questions:** *What is the tradeoff of a bigger or smaller block size?* (Smaller means less
waste but larger block tables and less efficient kernels.) *Recompute vs. swap preemption?*
*Why not just use `cudaMalloc` per request?*

---

## 4. Automatic prefix caching (APC)

When requests share a prefix (system prompt, RAG context, few-shot examples, earlier chat
turns), their KV blocks for that prefix are identical, so compute them once and reuse them.

**How vLLM does it:** each *full* block gets a hash of `(parent block's hash, the block's
token IDs, extra keys such as LoRA ID / multimodal hashes / cache salt)`. Chaining in the parent hash
means a block hash identifies the entire prefix up to that block. On a new request the
scheduler walks the prompt block by block, looks hashes up in a hash→block table and
reuses every hit. Freed blocks stay cached until evicted (LRU). Only full blocks are cached,
so the partial last block is always recomputed.

> **Correction to `plan.md`:** "radix tree" is **SGLang's** RadixAttention design. vLLM uses
> the **hash-based block table** above. Both achieve longest-prefix reuse with different data
> structures. Getting this right in an interview is a strong signal.

Effect: a cache hit skips prefill for the cached tokens, so **TTFT falls by roughly the cached
fraction of the prompt**. TPOT (decode) is unaffected, so end-to-end latency improves less when
outputs are long. Design implication: put **static content first** (system prompt, documents)
and variable content (the user question) last. Any change early in the prompt invalidates every
block after it.

How the benchmark guarantees a true cold start: a unique nonce in the first line changes block 0's
hash, and that changes every hash chained after it.

> Experiment: `test_cache.py` → `prefix_cache_ttft_<gpu>.png`, with caching on and with caching
> off as a control. Cached-token counts come from Prometheus (`vllm:prefix_cache_hits`).

**Likely questions:** *Why only full blocks?* *How do you route requests across replicas to
maximize cache hits?* (Prefix-aware / consistent-hash routing, e.g. the llm-d and Dynamo routers.)
*Security issue with shared caches?* (Timing side channels across tenants; `cache_salt` isolates them.)

---

## 5. Continuous batching

**Static batching** (e.g. `model.generate(batch)`) admits a batch, runs until the *longest*
sequence finishes, then admits the next batch. Short sequences finish early and leave their slots idle.

**Continuous (iteration-level) batching** (Orca, OSDI '22) makes scheduling decisions *every
decode step*. Finished sequences leave immediately and waiting requests join on the next step.
Combined with **chunked prefill**, new prompts are split into chunks and mixed into the same
step as ongoing decodes (bounded by `--max-num-batched-tokens`), so a long prompt arriving
doesn't stall everyone's token stream.

Why it pays off: decode is memory-bound (section 1), so a step costs about the same at
batch 1 or batch 32, because the weights are read once either way. Keeping the batch full turns
idle bandwidth into throughput nearly for free, until compute or KV capacity saturates.

Throughput/latency tradeoff: more concurrency gives higher aggregate tok/s but higher
per-user TPOT and queueing TTFT. The concurrency sweep shows the **knee**, which is where you set
`--max-num-seqs` or autoscaling thresholds for a latency SLO.

> Experiments: `kv_cache_demo.py` → `batch_decode_<gpu>.png` (step latency nearly flat with
> batch size). `load_test --scenario batching` → `batching_timeline_<gpu>.png` (static: sawtooth
> occupancy; continuous: saturated). `--scenario sweep` → `concurrency_sweep_<gpu>.png`.

**Likely questions:** *How does chunked prefill trade TTFT against TPOT?* *What is
prefill/decode disaggregation and when is it worth it?* *How would you autoscale on
`num_requests_waiting` vs. GPU utilization?* (GPU utilization is a poor signal for LLMs. Use queue
depth, KV usage and TTFT.)

---

## 6. Metrics glossary

| Metric | Definition | Dominated by |
|---|---|---|
| TTFT | submit → first token | queueing + prefill |
| TPOT / ITL | mean gap between output tokens | decode step time (batch size, context) |
| E2E latency | TTFT + TPOT × (n_out − 1) | output length |
| Throughput | output tokens/s across all requests | batch size × 1/step-time |
| Goodput | throughput of requests that *met the SLO* | what you actually sell |

Report percentiles (p50/p99), not means. Tail latency is what users notice.
