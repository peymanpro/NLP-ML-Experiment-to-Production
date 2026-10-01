from dataclasses import dataclass
from time import perf_counter


@dataclass
class InferenceMetrics:
    requests_total: int = 0
    errors_total: int = 0
    total_latency_ms: float = 0.0

    @property
    def average_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    def observe(self, latency_ms: float, *, error: bool = False) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        self.requests_total += 1
        self.total_latency_ms += latency_ms
        if error:
            self.errors_total += 1


class LatencyTimer:
    def __init__(self, metrics: InferenceMetrics) -> None:
        self.metrics = metrics
        self._start = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None:
        elapsed_ms = (perf_counter() - self._start) * 1000
        self.metrics.observe(elapsed_ms, error=exc_type is not None)
