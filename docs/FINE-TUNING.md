# Transformer Fine-Tuning Protocol

The fine-tuning track adapts a pretrained sequence-classification Transformer to the BANKING77 intent task.

## Training flow

~~~text
BANKING77
   ↓
Train / validation split
   ↓
Hugging Face tokenizer
   ↓
PyTorch tensors
   ↓
Pretrained Transformer
   ↓
AdamW
   ↓
Validation loss
   ↓
Best checkpoint
   ↓
Held-out evaluation
~~~

The first implementation intentionally uses direct PyTorch training rather than hiding the training loop inside a high-level trainer. This keeps optimization, validation, checkpoint selection, and device placement visible.

The test split is not used for checkpoint selection.

The runner supports bounded sample sizes for development runs. Final benchmark results must use a declared configuration and record the resulting artifacts.

The repository does not commit downloaded datasets or model weights.
