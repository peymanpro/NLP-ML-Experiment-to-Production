import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel


class TransformerClassifier(nn.Module):
    """Adapt a Hugging Face sequence classifier to the project's training contract."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = output.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError("transformer model did not return tensor logits")
        return logits
