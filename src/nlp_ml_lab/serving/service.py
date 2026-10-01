from dataclasses import dataclass, field
from typing import Protocol

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nlp_ml_lab.observability.metrics import InferenceMetrics


class TextPredictor(Protocol):
    model_version: str

    def predict(self, text: str) -> tuple[str, float]: ...


@dataclass
class InferenceService:
    predictor: TextPredictor | None = None
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    @property
    def model_version(self) -> str | None:
        return self.predictor.model_version if self.predictor else None

    def predict(self, text: str) -> tuple[str, float]:
        if not text.strip():
            raise ValueError("text must not be blank")
        if self.predictor is None:
            raise RuntimeError("model is not loaded")

        from time import perf_counter

        started = perf_counter()
        try:
            result = self.predictor.predict(text)
        except Exception:
            self.metrics.observe((perf_counter() - started) * 1000, error=True)
            raise

        self.metrics.observe((perf_counter() - started) * 1000)
        return result


@dataclass
class TorchTextPredictor:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    labels: list[str]
    model_version: str
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def predict(self, text: str) -> tuple[str, float]:
        self.model.eval()
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)[0]
            index = int(probabilities.argmax().item())
            confidence = float(probabilities[index].item())

        return self.labels[index], confidence
