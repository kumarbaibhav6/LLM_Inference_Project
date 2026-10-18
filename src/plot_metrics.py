"""Phase 5 - Charts + a markdown summary from everything in benchmarks/*.json.

    python -m src.plot_metrics            # uses the files for the current GPU
    python -m src.plot_metrics --gpu L4   # e.g. when plotting locally after downloading results

Each chart is skipped quietly if its input file has not been produced yet.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402

# Light-surface reference palette (categorical slots in fixed order; text never wears series colors).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10, "legend.frameon": False,
    "figure.dpi": 110,
})


def _load(name):
    path = config.BENCH_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, name, made):
    path = config.BENCH_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path.name)


def _bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)


# ------------------------------------------------------------------ charts

def plot_prefix_cache(gpu, made, md):
    on = _load(f"prefix_cache_cache_on_{gpu}.json")
    if not on:
        return
    off = _load(f"prefix_cache_cache_off_{gpu}.json")
    s_on = on["summary"]
    labels = [f"{r['prompt_tokens']}" for r in s_on]
    x = range(len(s_on))
    w = 0.27 if off else 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [("Query 1: cold start", [r["cold_ttft_ms"] for r in s_on], S1),
              ("Query 2: prefix-cache hit", [r["hit_ttft_ms"] for r in s_on], S2)]
    if off:
        series.append(("Query 2, caching disabled (control)", [r["hit_ttft_ms"] for r in off["summary"]], S3))
    offset = -(len(series) - 1) / 2
    for i, (name, vals, color) in enumerate(series):
        bars = ax.bar([xi + (offset + i) * w for xi in x], vals, w * 0.92, label=name, color=color)
        _bar_labels(ax, bars)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("System-prompt length (tokens)")
    ax.set_ylabel("Median time to first token (ms)")
    ax.set_title(f"Automatic prefix caching: TTFT, cold vs. cache hit ({gpu})")
    ax.legend(loc="upper left")
    _save(fig, f"prefix_cache_ttft_{gpu}.png", made)

    md.append("## Prefix caching (Phase 3)\n")
    md.append("| Prompt tokens | Cold TTFT (ms) | Hit TTFT (ms) | TTFT reduction | E2E reduction | Cached tokens |")
    md.append("|---|---|---|---|---|---|")
    for r in s_on:
        md.append(f"| {r['prompt_tokens']} | {r['cold_ttft_ms']} | {r['hit_ttft_ms']} | "
                  f"{r['ttft_reduction_pct']}% | {r['e2e_reduction_pct']}% | {r['hit_cached_tokens']} |")
    if off:
        worst = max(abs(r["ttft_reduction_pct"]) for r in off["summary"])
        md.append(f"\nControl (prefix caching disabled): second query TTFT within {worst}% of the first.")
    md.append("")


def plot_batching(gpu, made, md):
    d = _load(f"load_batching_{gpu}.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    conc = d["static"]["stats"]["concurrency"]
    for mode, color in (("static", S1), ("continuous", S2)):
        tl = d[mode]["timeline"]
        st = d[mode]["stats"]
        ax.step([s["t"] for s in tl], [s["running"] for s in tl], where="post", color=color,
                label=f"{mode} batching - {st['wall_s']:.1f} s, {st['output_tok_per_s']:.0f} tok/s")
    ax.axhline(conc, color=INK_2, lw=1, ls="--")
    ax.annotate(f"{conc} slots", (0, conc), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests running on the GPU")
    ax.set_title(f"Static vs. continuous batching, same 100 requests ({gpu})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _save(fig, f"batching_timeline_{gpu}.png", made)

    s, c = d["static"]["stats"], d["continuous"]["stats"]
    md.append("## Static vs. continuous batching (Phase 4)\n")
    md.append("| Schedule | Wall time (s) | Output tok/s | Mean running / slots | E2E p50 (s) |")
    md.append("|---|---|---|---|---|")
    for name, st in (("static", s), ("continuous", c)):
        md.append(f"| {name} | {st['wall_s']} | {st['output_tok_per_s']} | {st['mean_running']} / {conc} | {st['e2e_p50_s']} |")
    md.append(f"\nContinuous batching: **{s['wall_s'] / c['wall_s']:.2f}x** faster on identical requests.\n")


def plot_burst(gpu, md):
    d = _load(f"load_burst_{gpu}.json")
    if not d:
        return
    st = d["stats"]
    md.append("## 25-request burst (Phase 4)\n")
    md.append(f"- {st['requests']} concurrent requests, {st['failed']} failed, total {st['wall_s']} s "
              f"(single request alone: {st.get('single_request_s')} s)")
    md.append(f"- {st['output_tok_per_s']} output tok/s, TTFT p50 {st['ttft_p50_ms']} ms / p99 {st['ttft_p99_ms']} ms, "
              f"TPOT p50 {st['tpot_p50_ms']} ms, peak running {st['peak_running']}\n")


def plot_sweep(gpu, made, md):
    d = _load(f"load_sweep_{gpu}.json")
    if not d:
        return
    rows = d["rows"]
    c = [r["concurrency"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(c, [r["output_tok_per_s"] for r in rows], marker="o", color=S1)
    a1.set_title("Throughput")
    a1.set_ylabel("Output tokens / s")
    a2.plot(c, [r["tpot_p50_ms"] for r in rows], marker="o", color=S1)
    a2.set_title("Per-user decode latency")
    a2.set_ylabel("TPOT p50 (ms / token)")
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xticks(c, [str(v) for v in c])
        a.set_xlabel("Concurrent requests")
        a.set_ylim(bottom=0)
    fig.suptitle(f"Concurrency sweep, 256 output tokens/request ({gpu})", x=0.01, ha="left", fontweight="bold")
    _save(fig, f"concurrency_sweep_{gpu}.png", made)

    md.append("## Concurrency sweep\n")
    md.append("| Concurrency | Output tok/s | TPOT p50 (ms) | TTFT p99 (ms) | Peak KV usage |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['concurrency']} | {r['output_tok_per_s']} | {r['tpot_p50_ms']} | {r['ttft_p99_ms']} | {r['peak_kv_usage']} |")
    md.append("")


def plot_pressure(gpu, made, md):
    d = _load(f"load_pressure_starved_{gpu}.json") or _load(f"load_pressure_{gpu}.json")
    if not d:
        return
    tl, st = d["timeline"], d["stats"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    t = [s["t"] for s in tl]
    a1.step(t, [s["running"] for s in tl], where="post", color=S1, label="running")
    a1.step(t, [s["waiting"] for s in tl], where="post", color=S2, label="waiting (queued / preempted)")
    if st.get("contiguous_max_seqs"):
        a1.axhline(st["contiguous_max_seqs"], color=INK_2, lw=1, ls="--")
        a1.annotate(f"contiguous max-len allocation limit: {st['contiguous_max_seqs']}",
                    (0, st["contiguous_max_seqs"]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    a1.set_ylabel("Requests")
    fig.suptitle(f"PagedAttention under KV-cache pressure ({gpu})", x=0.01, ha="left", fontweight="bold")
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    a2.plot(t, [100 * s["kv_usage"] for s in tl], color=S1)
    a2.set_ylabel("KV cache used (%)")
    a2.set_xlabel("Time (s)")
    a2.set_ylim(0, 105)
    _save(fig, f"kv_pressure_{gpu}.png", made)

    md.append("## PagedAttention under KV pressure\n")
    md.append(f"- KV pool: {st.get('gpu_kv_cache_tokens')} tokens; reserving {config.MAX_MODEL_LEN} tokens per request "
              f"would admit **{st.get('contiguous_max_seqs')}** sequences")
    md.append(f"- vLLM ran up to **{st['peak_running']}** concurrently, peak KV usage {st['peak_kv_usage']}, "
              f"{st['preemptions']} preemptions, {st['failed']} failed requests\n")


def plot_kv_lab(gpu, made, md):
    d = _load(f"kv_cache_lab_{gpu}.json")
    if not d:
        return
    md.append("## KV-cache lab (raw transformers)\n")
    if "kv_math" in d:
        km = d["kv_math"]
        rows = km["schemes"]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        names = [r["scheme"].replace("_", " ") for r in rows]
        bars = ax.barh(names, [r["concurrent_seqs_in_budget"] for r in rows], color=S1, height=0.5)
        for b, r in zip(bars, rows):
            ax.annotate(f"{r['concurrent_seqs_in_budget']} seqs ({r['utilization_pct']}% used)",
                        (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.invert_yaxis()
        ax.set_xlabel(f"Concurrent sequences that fit in {km['budget_gb']} GB of KV cache")
        ax.set_title("KV allocation scheme vs. capacity (simulated chat traffic)")
        ax.set_xlim(0, max(r["concurrent_seqs_in_budget"] for r in rows) * 1.6)
        _save(fig, "kv_fragmentation.png", made)
        md.append(f"- KV cache: {km['bytes_per_token']:,} bytes/token "
                  f"(GQA saves {km['mha_bytes_per_token'] // km['bytes_per_token']}x vs. full MHA)")
        for r in rows:
            md.append(f"- {r['scheme']}: {r['utilization_pct']}% utilization, {r['concurrent_seqs_in_budget']} concurrent sequences")
    if "cache_vs_none" in d:
        cv = d["cache_vs_none"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(range(1, len(cv["no_cache_ms"]) + 1), cv["no_cache_ms"], color=S1, label="no KV cache (recompute all)")
        ax.plot(range(1, len(cv["with_cache_ms"]) + 1), cv["with_cache_ms"], color=S2, label="with KV cache")
        ax.set_xlabel("Generated token #")
        ax.set_ylabel("Step latency (ms)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Decode step latency with vs. without KV cache ({gpu})")
        ax.legend(loc="upper left")
        _save(fig, f"kv_cache_vs_none_{gpu}.png", made)
        a, b = sum(cv["with_cache_ms"]), sum(cv["no_cache_ms"])
        md.append(f"- {len(cv['with_cache_ms'])}-token generation: {a / 1e3:.2f} s with cache vs {b / 1e3:.2f} s without ({b / a:.1f}x)")
    if "prefill_decode" in d:
        rows = d["prefill_decode"]["rows"]
        n = [r["context"] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(n, [r["prefill_ms"] for r in rows], marker="o", color=S1)
        a1.set_title("Prefill (compute-bound)")
        a1.set_ylabel("Time for whole prompt (ms)")
        a2.plot(n, [r["decode_step_ms"] for r in rows], marker="o", color=S1)
        a2.set_title("One decode step (memory-bound)")
        a2.set_ylabel("Time per token (ms)")
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xticks(n, [str(v) for v in n])
            a.set_xlabel("Context length (tokens)")
            a.set_ylim(bottom=0)
        fig.suptitle(f"Prefill vs. decode cost by context length ({gpu})", x=0.01, ha="left", fontweight="bold")
        _save(fig, f"prefill_vs_decode_{gpu}.png", made)
    if "batch_decode" in d:
        rows = d["batch_decode"]["rows"]
        b = [r["batch"] for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(b, [r["decode_step_ms"] for r in rows], marker="o", color=S1)
        a1.set_title("Decode step latency")
        a1.set_ylabel("ms per step")
        a2.plot(b, [r["tok_per_s"] for r in rows], marker="o", color=S1)
        a2.set_title("Decode throughput")
        a2.set_ylabel("Tokens / s")
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xticks(b, [str(v) for v in b])
            a.set_xlabel("Batch size")
            a.set_ylim(bottom=0)
        fig.suptitle(f"Batching decode is nearly free: memory-bound regime ({gpu})", x=0.01, ha="left", fontweight="bold")
        _save(fig, f"batch_decode_{gpu}.png", made)
        md.append(f"- Decode batch 1 -> {b[-1]}: step latency {rows[0]['decode_step_ms']:.1f} -> "
                  f"{rows[-1]['decode_step_ms']:.1f} ms, throughput x{rows[-1]['tok_per_s'] / rows[0]['tok_per_s']:.0f}")
    md.append("")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default=None, help="GPU tag in result filenames (default: detect)")
    args = p.parse_args()
    gpu = args.gpu or config.gpu_tag()

    made, md = [], [f"# Benchmark summary - {config.MODEL} on {gpu}\n",
                    "_Generated by `python -m src.plot_metrics` from benchmarks/*.json._\n"]
    plot_kv_lab(gpu, made, md)
    plot_prefix_cache(gpu, made, md)
    plot_burst(gpu, md)
    plot_batching(gpu, made, md)
    plot_sweep(gpu, made, md)
    plot_pressure(gpu, made, md)

    summary = config.BENCH_DIR / f"summary_{gpu}.md"
    summary.write_text("\n".join(md))
    print("Charts:", ", ".join(made) or "(none - run the benchmarks first)")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
