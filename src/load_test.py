"""Phase 4 - Continuous batching under concurrent load.

Scenarios (all async, AsyncOpenAI + asyncio):

  burst     The plan's test: fire 25 requests at once, log each completion time
            and the batch's total time.
  batching  Same engine, two *client-side* schedules over the same 100 requests
            with mixed output lengths (32-512 tokens):
              static     - waves of 25; wave k+1 starts only when every request
                           in wave k is done (what HF `generate(batch)` does)
              continuous - keep 25 in flight; a new request enters the moment
                           any finishes (what the vLLM scheduler does per step)
            The gap between them is the idle time static batching wastes
            waiting on the longest sequence in each wave.
  sweep     Throughput / latency vs. concurrency (1 -> 128): shows that decode is
            memory-bound, so extra concurrent sequences are nearly free until
            compute or KV capacity saturates.
  pressure  128 concurrent requests. Run it against a server started with
            --num-gpu-blocks to starve the KV pool and watch PagedAttention
            admit far more sequences than contiguous allocation could, then
            preempt when the pool runs out.

A background task samples /metrics every 200 ms so every run comes with a
timeline of running/waiting requests and KV-cache usage.

    python -m src.load_test --scenario burst
    python -m src.load_test --scenario all
"""

import argparse
import asyncio
import json
import random
import time

import pandas as pd

from src import config
from src.client import (KV_USAGE, PREEMPTIONS, RUNNING, WAITING, async_client,
                        atimed_chat, metric, scrape_metrics)
from src.prompts import QUESTIONS, chat, system_prompt
from src.server import server_state


def run_async(coro):
    """asyncio.run() fails inside Jupyter/Colab's already-running loop; nest_asyncio fixes that."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import nest_asyncio
    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def make_requests(n: int, out_lens, prompt_tokens: int = 300, seed: int = 0):
    """n distinct prompts (unique nonce => no prefix sharing) with fixed output lengths."""
    rng = random.Random(seed)
    if isinstance(out_lens, int):
        lens = [out_lens] * n
    else:
        lens = [rng.randint(*out_lens) for _ in range(n)]
    return [(chat(system_prompt(prompt_tokens, nonce=f"req-{seed}-{i}"),
                  QUESTIONS[i % len(QUESTIONS)]), lens[i]) for i in range(n)]


class MetricsSampler:
    def __init__(self, interval: float = 0.2):
        self.interval, self.samples, self._task = interval, [], None

    async def _loop(self, t0):
        while True:
            try:
                m = await asyncio.to_thread(scrape_metrics)
                self.samples.append({
                    "t": time.perf_counter() - t0,
                    "running": metric(m, *RUNNING), "waiting": metric(m, *WAITING),
                    "kv_usage": metric(m, *KV_USAGE), "preemptions": metric(m, *PREEMPTIONS),
                })
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def start(self, t0):
        self._task = asyncio.create_task(self._loop(t0))

    async def stop(self):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


async def _execute(reqs, mode: str, concurrency: int):
    """Run `reqs` under a schedule. Returns (results, wall_time, metric samples)."""
    client = async_client()
    sampler = MetricsSampler()
    t0 = time.perf_counter()
    sampler.start(t0)

    async def one(i, msgs, n_out):
        return await atimed_chat(client, msgs, max_tokens=n_out, ignore_eos=True, request_id=i)

    if mode == "static":
        results = []
        for w in range(0, len(reqs), concurrency):
            wave = reqs[w:w + concurrency]
            results += await asyncio.gather(*(one(w + j, m, n) for j, (m, n) in enumerate(wave)))
    else:  # continuous: a semaphore keeps exactly `concurrency` requests in flight
        sem = asyncio.Semaphore(concurrency)

        async def gated(i, m, n):
            async with sem:
                return await one(i, m, n)
        results = await asyncio.gather(*(gated(i, m, n) for i, (m, n) in enumerate(reqs)))

    wall = time.perf_counter() - t0
    await asyncio.sleep(0.5)  # one more sample after the drain
    await sampler.stop()
    await client.close()  # otherwise httpx tries to close sockets after the loop is gone
    for r in results:
        r.t_submit -= t0
    return results, wall, sampler.samples


def _stats(results, wall, samples, concurrency) -> dict:
    ok = [r for r in results if not r.error]
    df = pd.DataFrame([r.to_dict() for r in ok])
    out_tok = int(df.completion_tokens.sum()) if len(df) else 0
    running = [s["running"] for s in samples if s["t"] <= wall]
    return {
        "concurrency": concurrency,
        "requests": len(results), "failed": len(results) - len(ok),
        "wall_s": round(wall, 3),
        "output_tok_per_s": round(out_tok / wall, 1),
        "req_per_s": round(len(ok) / wall, 3),
        "ttft_p50_ms": round(df.ttft.median() * 1e3, 1) if len(df) else None,
        "ttft_p99_ms": round(df.ttft.quantile(0.99) * 1e3, 1) if len(df) else None,
        "tpot_p50_ms": round(df.tpot.median() * 1e3, 2) if len(df) else None,
        "tpot_p99_ms": round(df.tpot.quantile(0.99) * 1e3, 2) if len(df) else None,
        "e2e_p50_s": round(df.e2e.median(), 3) if len(df) else None,
        "mean_running": round(sum(running) / len(running), 2) if running else None,
        "peak_running": max(running, default=None),
        "peak_waiting": max((s["waiting"] for s in samples), default=None),
        "peak_kv_usage": max((s["kv_usage"] for s in samples), default=None),
        "preemptions": (samples[-1]["preemptions"] - samples[0]["preemptions"]) if samples else None,
    }


def _record(results):
    return [r.to_dict() for r in results]


# ------------------------------------------------------------------ scenarios

def burst(n=25, out_len=256):
    reqs = make_requests(n, out_len, seed=1)
    results, wall, samples = run_async(_execute(reqs, "continuous", concurrency=n))
    print(f"Burst: {n} concurrent requests, {out_len} output tokens each")
    for r in sorted(results, key=lambda r: r.e2e):
        status = "ERROR " + r.error if r.error else f"TTFT {r.ttft*1e3:7.1f} ms  done {r.e2e:6.2f} s  ({r.completion_tokens} tok)"
        print(f"  req {r.request_id:>3}: {status}")
    st = _stats(results, wall, samples, n)
    print(f"Total batch time: {wall:.2f} s | {st['output_tok_per_s']} output tok/s | "
          f"peak running {st['peak_running']} | failed {st['failed']}")
    # Compare against the time one request takes alone: with continuous batching
    # the whole burst should take only modestly longer than a single request.
    solo, solo_wall, _ = run_async(_execute(reqs[:1], "continuous", 1))
    st["single_request_s"] = round(solo_wall, 3)
    print(f"Single request alone: {solo_wall:.2f} s -> burst served {n} requests in "
          f"{wall / solo_wall:.1f}x the time of one ({n * solo_wall / wall:.1f}x effective speed-up vs. serial)")
    return {"stats": st, "requests": _record(results), "timeline": samples}


def batching(n=100, concurrency=25, out_range=(32, 512)):
    reqs = make_requests(n, out_range, seed=2)
    out = {}
    for mode in ("static", "continuous"):
        results, wall, samples = run_async(_execute(reqs, mode, concurrency))
        st = _stats(results, wall, samples, concurrency)
        out[mode] = {"stats": st, "requests": _record(results), "timeline": samples}
        print(f"{mode:>10}: wall {st['wall_s']:6.2f} s | {st['output_tok_per_s']:7.1f} tok/s | "
              f"mean running {st['mean_running']} / {concurrency} | e2e p50 {st['e2e_p50_s']} s")
    s, c = out["static"]["stats"], out["continuous"]["stats"]
    print(f"Continuous batching: {s['wall_s'] / c['wall_s']:.2f}x faster, "
          f"{c['output_tok_per_s'] / s['output_tok_per_s']:.2f}x throughput on identical requests")
    return out


def sweep(levels=(1, 2, 4, 8, 16, 32, 64, 128), out_len=256):
    rows = []
    for c in levels:
        reqs = make_requests(max(2 * c, 16), out_len, seed=3)
        results, wall, samples = run_async(_execute(reqs, "continuous", c))
        st = _stats(results, wall, samples, c)
        rows.append(st)
        print(f"  concurrency {c:>4}: {st['output_tok_per_s']:8.1f} tok/s | TPOT p50 {st['tpot_p50_ms']:6.2f} ms | "
              f"TTFT p99 {st['ttft_p99_ms']:8.1f} ms | peak KV {st['peak_kv_usage']}")
    return {"rows": rows}


def pressure(n=128, out_len=384, prompt_tokens=600):
    state = server_state()
    reqs = make_requests(n, out_len, prompt_tokens=prompt_tokens, seed=4)
    results, wall, samples = run_async(_execute(reqs, "continuous", n))
    st = _stats(results, wall, samples, n)
    kv_tokens = state.get("gpu_kv_cache_tokens")
    st["gpu_kv_cache_tokens"] = kv_tokens
    st["contiguous_max_seqs"] = (kv_tokens // config.MAX_MODEL_LEN) if kv_tokens else None
    print(f"KV pool: {kv_tokens} tokens -> contiguous max-len allocation would admit "
          f"{st['contiguous_max_seqs']} sequences")
    print(f"PagedAttention admitted up to {st['peak_running']} concurrently | "
          f"peak KV usage {st['peak_kv_usage']} | preemptions {st['preemptions']} | failed {st['failed']}")
    return {"stats": st, "requests": _record(results), "timeline": samples}


SCENARIOS = {"burst": burst, "batching": batching, "sweep": sweep, "pressure": pressure}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="burst")
    p.add_argument("--tag", default="", help="Suffix for the output file, e.g. 'starved'")
    args = p.parse_args()

    names = ["burst", "batching", "sweep"] if args.scenario == "all" else [args.scenario]
    config.BENCH_DIR.mkdir(exist_ok=True)
    for name in names:
        print(f"\n=== {name} ===")
        result = SCENARIOS[name]()
        suffix = f"_{args.tag}" if args.tag else ""
        path = config.BENCH_DIR / f"load_{name}{suffix}_{config.gpu_tag()}.json"
        path.write_text(json.dumps({"server": server_state(), **result}, indent=2))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
