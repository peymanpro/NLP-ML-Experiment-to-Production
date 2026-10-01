from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingResult:
    loss: float
    samples: int


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> TrainingResult:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("training loader produced no samples")

    return TrainingResult(
        loss=total_loss / total_samples,
        samples=total_samples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
) -> TrainingResult:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += float(loss) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("evaluation loader produced no samples")

    return TrainingResult(
        loss=total_loss / total_samples,
        samples=total_samples,
    )
