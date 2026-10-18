"""Timed streaming requests and Prometheus scraping against the vLLM server.

TTFT can only be measured with streaming: the non-streaming API returns once
the whole completion is done, which hides the prefill/decode split entirely.
"""

import re
import time
from dataclasses import asdict, dataclass, field

import requests
from openai import AsyncOpenAI, OpenAI

from src import config


@dataclass
class RequestResult:
    request_id: int = 0
    t_submit: float = 0.0        # wall-clock (perf_counter) when the request was sent
    ttft: float = 0.0            # seconds to first content token  (~ queueing + prefill)
    e2e: float = 0.0             # seconds to last token
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_times: list = field(default_factory=list, repr=False)  # arrival time of each chunk
    error: str = ""

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first (the decode rate one user sees)."""
        if self.completion_tokens <= 1:
            return 0.0
        return (self.e2e - self.ttft) / (self.completion_tokens - 1)

    def to_dict(self, keep_token_times: bool = False) -> dict:
        d = asdict(self)
        if not keep_token_times:
            d.pop("token_times")
        d["tpot"] = self.tpot
        return d


def sync_client() -> OpenAI:
    return OpenAI(base_url=config.API_URL, api_key=config.API_KEY, timeout=600)


def async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=config.API_URL, api_key=config.API_KEY, timeout=600)


def _request_kwargs(messages, max_tokens, ignore_eos):
    kw = dict(
        model=config.MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    if ignore_eos:
        # vLLM extension: generate exactly max_tokens. Makes output length a
        # controlled variable instead of depending on what the model says.
        kw["extra_body"] = {"ignore_eos": True}
    return kw


def _consume_chunk(chunk, res: RequestResult, t0: float):
    now = time.perf_counter()
    if chunk.usage:
        res.prompt_tokens = chunk.usage.prompt_tokens
        res.completion_tokens = chunk.usage.completion_tokens
    if chunk.choices and chunk.choices[0].delta.content:
        if not res.token_times:
            res.ttft = now - t0
        res.token_times.append(now - t0)


def timed_chat(client: OpenAI, messages, max_tokens=64, ignore_eos=False, request_id=0) -> RequestResult:
    res = RequestResult(request_id=request_id)
    t0 = res.t_submit = time.perf_counter()
    try:
        for chunk in client.chat.completions.create(**_request_kwargs(messages, max_tokens, ignore_eos)):
            _consume_chunk(chunk, res, t0)
    except Exception as e:  # keep the benchmark going; failures are reported
        res.error = repr(e)
    res.e2e = time.perf_counter() - t0
    return res


async def atimed_chat(client: AsyncOpenAI, messages, max_tokens=64, ignore_eos=False, request_id=0) -> RequestResult:
    res = RequestResult(request_id=request_id)
    t0 = res.t_submit = time.perf_counter()
    try:
        stream = await client.chat.completions.create(**_request_kwargs(messages, max_tokens, ignore_eos))
        async for chunk in stream:
            _consume_chunk(chunk, res, t0)
    except Exception as e:
        res.error = repr(e)
    res.e2e = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------- Prometheus

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def scrape_metrics() -> dict[str, float]:
    """Parse /metrics into {name: value}, summing across label sets.

    Counter names are normalised by stripping the `_total` suffix, so
    `vllm:prefix_cache_hits_total` is available as `vllm:prefix_cache_hits`.
    """
    text = requests.get(f"{config.BASE_URL}/metrics", timeout=5).text
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group(1).removesuffix("_total")
        try:
            out[name] = out.get(name, 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def metric(snapshot: dict, *names: str, default: float = 0.0) -> float:
    """First metric present among `names` (metric names drift across vLLM versions)."""
    for n in names:
        if n in snapshot:
            return snapshot[n]
    return default


KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREFIX_HITS = ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits")
PREFIX_QUERIES = ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")
PREEMPTIONS = ("vllm:num_preemptions",)
RUNNING = ("vllm:num_requests_running",)
WAITING = ("vllm:num_requests_waiting",)
