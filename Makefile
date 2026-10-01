PYTHON ?= python

.PHONY: test lint format-check typecheck quality baseline pytorch transformer finetune retrieval serve

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

format-check:
	ruff format --check .

typecheck:
	mypy src

quality: test lint format-check typecheck

baseline:
	$(PYTHON) scripts/run_banking77_baseline.py

pytorch:
	$(PYTHON) scripts/run_pytorch_baseline.py

transformer:
	$(PYTHON) scripts/run_transformer_baseline.py

finetune:
	$(PYTHON) scripts/run_transformer_finetuning.py

retrieval:
	$(PYTHON) scripts/run_scifact_retrieval.py

serve:
	$(PYTHON) scripts/serve.py
