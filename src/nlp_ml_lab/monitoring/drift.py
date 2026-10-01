from collections import Counter
from math import log2


def categorical_distribution(values: list[str]) -> dict[str, float]:
    if not values:
        raise ValueError("values must not be empty")
    counts = Counter(values)
    total = len(values)
    return {key: count / total for key, count in counts.items()}


def jensen_shannon_divergence(
    reference: list[str],
    current: list[str],
) -> float:
    if not reference or not current:
        raise ValueError("reference and current values must not be empty")

    reference_dist = categorical_distribution(reference)
    current_dist = categorical_distribution(current)
    keys = set(reference_dist) | set(current_dist)

    midpoint = {
        key: (reference_dist.get(key, 0.0) + current_dist.get(key, 0.0)) / 2
        for key in keys
    }

    def kl_divergence(
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        value = 0.0
        for key in keys:
            probability = left.get(key, 0.0)
            if probability == 0.0:
                continue
            midpoint_probability = right.get(key, 0.0)
            safe_probability = midpoint_probability if midpoint_probability > 0 else 1e-12
            value += probability * log2(probability / safe_probability)
        return value

    return float(
        0.5 * kl_divergence(reference_dist, midpoint)
        + 0.5 * kl_divergence(current_dist, midpoint)
    )
