from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


@dataclass(frozen=True)
class HuggingFaceBundle:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel


def load_sequence_classifier(
    model_id: str,
    *,
    num_labels: int,
) -> HuggingFaceBundle:
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if num_labels <= 1:
        raise ValueError("num_labels must be greater than one")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return HuggingFaceBundle(tokenizer=tokenizer, model=model)
