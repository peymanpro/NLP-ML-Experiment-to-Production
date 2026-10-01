# Classical Baseline Protocol

The first modeling stage deliberately uses a sparse lexical representation and a linear classifier.

Pipeline:

~~~text
Raw text
  ↓
Conservative normalization
  ↓
TF-IDF unigram + bigram features
  ↓
Logistic Regression
  ↓
Validation metrics
~~~

The baseline is intentionally simple. Its role is not to compete with a Transformer; it establishes a reference point against which additional representation and model complexity can be measured.

The baseline must be trained only on the training split. The validation set is used for model development, while the final test split remains held out until the final evaluation phase.

The implementation currently provides the model and evaluation infrastructure. Full BANKING77 benchmark execution belongs to the dataset-ingestion and experiment phases.
