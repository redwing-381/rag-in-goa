"""Span timing. One helper, used by every stage, so the trace is never partial.

Timing is recorded in a `finally` block: a stage that raises still reports how
long it burned, which is exactly the case you need when diagnosing a deadline
overrun.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from ragoa.schemas import Trace


@contextmanager
def timed(trace: Trace, name: str, **metadata):
    start = time.perf_counter()
    try:
        yield
    finally:
        trace.add(name, (time.perf_counter() - start) * 1000.0, **metadata)


class Stopwatch:
    """Manual timer for code that cannot be wrapped in a context manager."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def lap_ms(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self.start) * 1000.0
        self.start = now
        return elapsed

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0
