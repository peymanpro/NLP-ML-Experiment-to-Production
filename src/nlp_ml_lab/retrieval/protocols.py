from typing import Protocol


class DenseSearcher(Protocol):
    def search(self, query: str, *, k: int = 10) -> list[tuple[str, float]]:
        ...
