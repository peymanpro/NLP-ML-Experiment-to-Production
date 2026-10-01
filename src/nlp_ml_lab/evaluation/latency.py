from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def summarize_latencies(values_ms: list[float]) -> LatencySummary:
    if not values_ms:
        raise ValueError("latencies must not be empty")
    if any(value < 0 for value in values_ms):
        raise ValueError("latencies must be non-negative")

    ordered = sorted(values_ms)

    def percentile(percent: float) -> float:
        index = min(
            len(ordered) - 1,
            max(0, round((percent / 100) * (len(ordered) - 1))),
        )
        return float(ordered[index])

    return LatencySummary(
        count=len(ordered),
        mean_ms=float(sum(ordered) / len(ordered)),
        p50_ms=percentile(50),
        p95_ms=percentile(95),
    )
