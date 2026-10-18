"""Phase 0 - KV-cache lab (raw HF transformers, no vLLM).

Run this BEFORE starting the vLLM server: both want most of the GPU.

  1. kv_math        Bytes of KV per token for the model, and what contiguous vs.
                    paged allocation does to a fixed memory budget (simulation).
  2. cache_vs_none  Per-step decode latency with and without a KV cache. Without
                    it every step re-runs the whole sequence: O(N) work per token,
                    O(N^2) for the generation.
  3. prefill_decode Prefill time vs. prompt length (compute-bound, grows with N)
                    vs. one decode step at the same context (memory-bound, ~flat).
  4. batch_decode   One decode step at batch 1..64. Nearly flat => decode is
                    bandwidth-bound and batching is close to free throughput,
                    which is the whole economic argument for continuous batching.

    python -m src.kv_cache_demo                 # all experiments
    python -m src.kv_cache_demo --only kv_math  # CPU-only, no model download beyond config
"""

import argparse
import json
import math
import random
import statistics
import time

from src import config


# ------------------------------------------------------------------ 1. KV math

def kv_bytes_per_token(cfg, dtype_bytes: int = 2) -> int:
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # 2 tensors (K and V) x layers x kv_heads x head_dim x bytes
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes


def kv_math(budget_gb: float = 8.0, block_size: int = 16, n_seqs: int = 2000, seed: int = 0) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(config.MODEL)
    bpt = kv_bytes_per_token(cfg)
    max_len = config.MAX_MODEL_LEN
    print(f"{config.MODEL}: {cfg.num_hidden_layers} layers, {cfg.num_attention_heads} q-heads, "
          f"{cfg.num_key_value_heads} kv-heads (GQA)")
    print(f"KV cache: {bpt:,} bytes/token = {bpt / 1024:.1f} KiB/token; "
          f"one {max_len}-token sequence = {bpt * max_len / 2**20:.0f} MiB")
    mha_bpt = bpt * cfg.num_attention_heads // cfg.num_key_value_heads
    print(f"(Without GQA it would be {mha_bpt / 1024:.1f} KiB/token - "
          f"{cfg.num_attention_heads // cfg.num_key_value_heads}x more.)")

    # Realistic chat traffic: most requests are far shorter than max_model_len.
    rng = random.Random(seed)
    lens = []
    for _ in range(n_seqs):
        prompt = int(rng.lognormvariate(math.log(400), 0.8))
        output = int(rng.lognormvariate(math.log(200), 0.7))
        lens.append(max(16, min(prompt + output, max_len)))
    used = sum(lens)

    schemes = {
        # Pre-vLLM servers (e.g. FasterTransformer / naive HF): reserve max_model_len per request.
        "contiguous_max_len": [max_len] * n_seqs,
        # Orca-style: reserve prompt + max_tokens (here 1024) up front.
        "contiguous_prompt_plus_1024": [min(L + 1024, max_len) for L in lens],
        # PagedAttention: fixed-size blocks allocated on demand; waste < 1 block per sequence.
        f"paged_block_{block_size}": [math.ceil(L / block_size) * block_size for L in lens],
    }
    budget_tokens = int(budget_gb * 2**30 / bpt)
    rows = []
    for name, reserved in schemes.items():
        total = sum(reserved)
        rows.append({
            "scheme": name,
            "utilization_pct": round(100 * used / total, 1),
            "wasted_pct": round(100 * (1 - used / total), 1),
            "concurrent_seqs_in_budget": int(budget_tokens / (total / n_seqs)),
        })
    print(f"\nSimulated {n_seqs} requests (mean {used / n_seqs:.0f} tokens), {budget_gb} GB KV budget "
          f"= {budget_tokens:,} tokens:")
    for r in rows:
        print(f"  {r['scheme']:<30} utilization {r['utilization_pct']:5.1f}%  "
              f"-> {r['concurrent_seqs_in_budget']:>5} concurrent sequences")
    return {"bytes_per_token": bpt, "mha_bytes_per_token": mha_bpt, "budget_gb": budget_gb,
            "mean_seq_len": used / n_seqs, "schemes": rows}


# ------------------------------------------------------------ GPU experiments

def _load():
    import torch
    from transformers import AutoModelForCausalLM

    major, _ = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if major >= 8 else torch.float16  # T4 (SM 7.5) has no bf16
    model = AutoModelForCausalLM.from_pretrained(config.MODEL, dtype=dtype, attn_implementation="sdpa")
    return model.to("cuda").eval(), torch


def _timed(torch, fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def cache_vs_none(model, torch, prompt_len=1536, new_tokens=256) -> dict:
    # A long prompt is needed: raw HF forward passes carry ~30 ms of fixed
    # kernel-launch overhead, which hides recompute cost at short contexts.
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device="cuda")
    res = {}
    with torch.inference_mode():
        # With KV cache: each step feeds ONE token and attends to stored K/V.
        out = model(ids, use_cache=True, logits_to_keep=1)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        steps = []
        for _ in range(new_tokens):
            out, dt = _timed(torch, lambda: model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1))
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            steps.append(dt * 1e3)
        res["with_cache_ms"] = steps
        past = None  # free the cache before the no-cache run

        # Without: every step recomputes K/V for the entire sequence so far.
        seq, steps = ids, []
        for _ in range(new_tokens):
            out, dt = _timed(torch, lambda: model(seq, use_cache=False, logits_to_keep=1))
            seq = torch.cat([seq, out.logits[:, -1:].argmax(-1)], dim=1)
            steps.append(dt * 1e3)
        res["no_cache_ms"] = steps
    a, b = sum(res["with_cache_ms"]), sum(res["no_cache_ms"])
    print(f"Generate {new_tokens} tokens: with cache {a / 1e3:.2f} s, without {b / 1e3:.2f} s ({b / a:.1f}x slower). "
          f"Last step: {res['with_cache_ms'][-1]:.1f} ms vs {res['no_cache_ms'][-1]:.1f} ms")
    return res


def prefill_decode(model, torch, lengths=(128, 256, 512, 1024, 2048, 4096), reps=3) -> dict:
    rows = []
    with torch.inference_mode():
        for n in lengths:
            ids = torch.randint(0, model.config.vocab_size, (1, n), device="cuda")
            pre = []
            for _ in range(reps):
                out, dt = _timed(torch, lambda: model(ids, use_cache=True, logits_to_keep=1))
                pre.append(dt)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            dec = []
            for _ in range(10):
                out, dt = _timed(torch, lambda: model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1))
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                dec.append(dt)
            row = {"context": n, "prefill_ms": statistics.median(pre) * 1e3,
                   "decode_step_ms": statistics.median(dec) * 1e3}
            row["prefill_tok_per_s"] = n / (row["prefill_ms"] / 1e3)
            rows.append(row)
            print(f"  context {n:>5}: prefill {row['prefill_ms']:8.1f} ms ({row['prefill_tok_per_s']:9.0f} tok/s) | "
                  f"decode step {row['decode_step_ms']:6.2f} ms")
            past = out = None
    return {"rows": rows}


def batch_decode(model, torch, batches=(1, 4, 16, 32, 64), context=512) -> dict:
    rows = []
    with torch.inference_mode():
        for b in batches:
            ids = torch.randint(0, model.config.vocab_size, (b, context), device="cuda")
            out = model(ids, use_cache=True, logits_to_keep=1)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            dec = []
            for _ in range(10):
                out, dt = _timed(torch, lambda: model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1))
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                dec.append(dt)
            step = statistics.median(dec) * 1e3
            rows.append({"batch": b, "decode_step_ms": step, "tok_per_s": b / (step / 1e3)})
            print(f"  batch {b:>3}: decode step {step:6.2f} ms -> {rows[-1]['tok_per_s']:8.0f} tok/s")
            past = out = None
            torch.cuda.empty_cache()
    return {"context": context, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["kv_math", "cache_vs_none", "prefill_decode", "batch_decode"])
    args = p.parse_args()

    config.BENCH_DIR.mkdir(exist_ok=True)
    out = config.BENCH_DIR / f"kv_cache_lab_{config.gpu_tag()}.json"
    # Merge so that `--only X` re-runs one experiment without dropping the others.
    results = json.loads(out.read_text()) if out.exists() else {}
    results.update({"model": config.MODEL, "gpu": config.gpu_info()[0]})
    if args.only in (None, "kv_math"):
        print("=== 1. KV-cache memory math ===")
        results["kv_math"] = kv_math()
    if args.only != "kv_math":
        model, torch = _load()
        for name, fn in (("cache_vs_none", cache_vs_none), ("prefill_decode", prefill_decode),
                         ("batch_decode", batch_decode)):
            if args.only in (None, name):
                print(f"\n=== {name} ===")
                results[name] = fn(model, torch)

    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
