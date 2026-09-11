"""CUDA-correct timing primitives.

CUDA kernel launches are asynchronous: without an explicit sync you are timing
how long it takes the CPU to *enqueue* work, not how long the GPU takes to do it.
That mistake silently produces numbers that are too good and internally consistent,
which is the worst kind of wrong. Everything in this project times through here.
(See d2l ch. 13.2.)
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

import torch


@contextmanager
def cuda_timer(store: list[float]):
    """Append the GPU-side elapsed ms of the enclosed block to `store`.

    Uses CUDA events rather than perf_counter: events are recorded *in the stream*,
    so they measure device execution without forcing a full-device synchronize.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    end.synchronize()
    store.append(start.elapsed_time(end))


@contextmanager
def wall_timer(store: list[float]):
    """Append wall-clock ms, syncing first so pending GPU work is included."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    store.append((time.perf_counter() - t0) * 1000.0)


def summarize(samples: list[float]) -> dict[str, float]:
    """Median + IQR, not mean + stdev.

    Decode latency on a desktop GPU has a heavy right tail (display compositing,
    background apps, clock throttling). The mean tracks that tail; the median
    tracks the behaviour we are actually trying to characterize.
    """
    if not samples:
        return {}
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "n": n,
        "median": statistics.median(ordered),
        "p25": ordered[max(0, int(0.25 * n) - 1)],
        "p75": ordered[min(n - 1, int(0.75 * n))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def model_weight_bytes(model: torch.nn.Module) -> int:
    """Bytes of parameters that must stream from HBM for one forward pass."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
