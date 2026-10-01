from nlp_ml_lab.text.vocabulary import build_vocabulary


def test_vocabulary_order_is_deterministic() -> None:
    vocab = build_vocabulary(["card cash", "cash card"])

    assert vocab.token_to_id["<pad>"] == 0
    assert vocab.token_to_id["<unk>"] == 1
    assert vocab.token_to_id["card"] < vocab.token_to_id["cash"]


def test_unknown_tokens_use_unk_id() -> None:
    vocab = build_vocabulary(["known token"])

    encoded = vocab.encode("known unseen")

    assert encoded[0] == vocab.token_to_id["known"]
    assert encoded[1] == vocab.unk_id
