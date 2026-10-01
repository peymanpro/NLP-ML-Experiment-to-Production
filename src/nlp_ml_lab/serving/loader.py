import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nlp_ml_lab.serving.service import TorchTextPredictor


def load_predictor_from_environment() -> TorchTextPredictor:
    model_path = os.getenv("NLP_ML_MODEL_PATH", "").strip()
    model_version = os.getenv("NLP_ML_MODEL_VERSION", "dev").strip() or "dev"

    if not model_path:
        raise RuntimeError("NLP_ML_MODEL_PATH is not configured")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    labels_path = path / "labels.json"
    if labels_path.exists():
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise TypeError("labels.json must contain a list of strings")
        labels = payload
    else:
        id_to_label = getattr(model.config, "id2label", {})
        labels = [str(id_to_label[index]) for index in range(model.config.num_labels)]

    return TorchTextPredictor(
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        model_version=model_version,
        device=torch.device("cpu"),
    )
