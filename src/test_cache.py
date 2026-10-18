"""Phase 3 - Automatic Prefix Caching: cold vs. cache-hit TTFT.

For each prefix length and trial:
  Query 1 (cold):  fresh system prompt (unique nonce in the first block) + question A
  Query 2 (hit):   the *same* system prompt                               + question B

Query 2 shares every full KV block of the system prompt with Query 1, so vLLM
skips their prefill and only computes the question's tokens. Run once with the
server's prefix caching ON and once OFF: the OFF run is the control that proves
the speed-up comes from the cache, not from warm-up or OS effects.

    python -m src.test_cache                    # tag inferred from server config
    python -m src.test_cache --lengths 800 --trials 1   # the plan's single-shot version
"""

import argparse
import json
import time

import pandas as pd

from src import config
from src.client import PREFIX_HITS, PREFIX_QUERIES, metric, scrape_metrics, sync_client, timed_chat
from src.prompts import QUESTIONS, chat, system_prompt
from src.server import server_state


def warmup(client):
    # The very first request pays one-off costs (CUDA graph capture is done at
    # startup, but tokenizer/detokenizer, HTTP pools and allocator warm-up are not).
    for q in QUESTIONS[:2]:
        timed_chat(client, chat("You are a helpful assistant.", q), max_tokens=8)


def run(lengths, trials, max_tokens):
    client = sync_client()
    warmup(client)
    rows = []
    for L in lengths:
        for t in range(trials):
            sys_prompt = system_prompt(L, nonce=f"{time.time_ns()}")
            pair = []
            for kind, q in (("cold", QUESTIONS[t % len(QUESTIONS)]),
                            ("hit", QUESTIONS[(t + 1) % len(QUESTIONS)])):
                before = scrape_metrics()
                # Fixed output length: different questions would otherwise produce
                # different answer lengths and confound the end-to-end comparison.
                r = timed_chat(client, chat(sys_prompt, q), max_tokens=max_tokens, ignore_eos=True)
                after = scrape_metrics()
                hits = metric(after, *PREFIX_HITS) - metric(before, *PREFIX_HITS)
                queries = metric(after, *PREFIX_QUERIES) - metric(before, *PREFIX_QUERIES)
                if r.error:
                    raise RuntimeError(f"Request failed: {r.error}")
                rows.append({
                    "target_len": L, "trial": t, "query": kind,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "ttft_ms": r.ttft * 1e3, "e2e_ms": r.e2e * 1e3, "tpot_ms": r.tpot * 1e3,
                    "cached_tokens": int(hits), "queried_tokens": int(queries),
                })
                pair.append(r)
            print(f"len~{L:>5} trial {t}: prompt={pair[0].prompt_tokens:>5} tok | "
                  f"cold TTFT {pair[0].ttft*1e3:7.1f} ms | hit TTFT {pair[1].ttft*1e3:7.1f} ms | "
                  f"cached {rows[-1]['cached_tokens']}/{rows[-1]['queried_tokens']} tok")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["target_len", "query"]).agg(
        prompt_tokens=("prompt_tokens", "median"),
        ttft_ms=("ttft_ms", "median"),
        e2e_ms=("e2e_ms", "median"),
        cached_tokens=("cached_tokens", "median"),
    ).unstack("query")
    s = pd.DataFrame({
        "prompt_tokens": g["prompt_tokens"]["cold"].astype(int),
        "cold_ttft_ms": g["ttft_ms"]["cold"],
        "hit_ttft_ms": g["ttft_ms"]["hit"],
        "cold_e2e_ms": g["e2e_ms"]["cold"],
        "hit_e2e_ms": g["e2e_ms"]["hit"],
        "hit_cached_tokens": g["cached_tokens"]["hit"].astype(int),
    })
    s["ttft_reduction_pct"] = 100 * (1 - s.hit_ttft_ms / s.cold_ttft_ms)
    s["e2e_reduction_pct"] = 100 * (1 - s.hit_e2e_ms / s.cold_e2e_ms)
    return s.round(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+", default=[750, 1500, 2500, 3500],
                   help="Approximate system-prompt lengths in tokens (max_model_len is 4096)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--tag", default=None, help="Defaults to cache_on / cache_off from server config")
    args = p.parse_args()

    state = server_state()
    tag = args.tag or ("cache_on" if state.get("prefix_caching", True) else "cache_off")
    print(f"Server: {state.get('gpu', '?')} | prefix caching = {state.get('prefix_caching')} | tag = {tag}\n")

    df = run(args.lengths, args.trials, args.max_tokens)
    summary = summarize(df)
    print("\nMedian over trials:")
    print(summary.to_string())

    # The plan's headline number: one heavy (>500-word) prompt, cold vs. hit.
    head = summary.iloc[(summary.prompt_tokens - 800).abs().argmin()]
    print(f"\n>>> {int(head.prompt_tokens)} token system prompt: TTFT {head.cold_ttft_ms:.1f} ms -> "
          f"{head.hit_ttft_ms:.1f} ms ({head.ttft_reduction_pct:.1f}% reduction), "
          f"total latency {head.e2e_reduction_pct:.1f}% reduction")
    longest = summary.iloc[-1]
    print(f">>> {int(longest.prompt_tokens)} token system prompt: {longest.ttft_reduction_pct:.1f}% TTFT reduction")

    if tag == "cache_on" and longest.ttft_reduction_pct < 30:
        print("WARNING: expected a large TTFT drop with prefix caching on. Check the server flags.")
    if tag == "cache_off" and summary.hit_cached_tokens.max() > 0:
        print("WARNING: cache hits recorded although prefix caching is supposed to be off.")

    config.BENCH_DIR.mkdir(exist_ok=True)
    out = config.BENCH_DIR / f"prefix_cache_{tag}_{config.gpu_tag()}.json"
    out.write_text(json.dumps({
        "server": state, "tag": tag, "max_tokens": args.max_tokens,
        "summary": summary.reset_index().to_dict(orient="records"),
        "raw": df.to_dict(orient="records"),
    }, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
