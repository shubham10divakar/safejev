import importlib.util
import os
import string

import pytest

HAS_MODEL = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))


def pytest_collection_modifyitems(config, items):
    if HAS_MODEL:
        return
    skip = pytest.mark.skip(reason="needs torch + transformers + peft")
    for it in items:
        if "model" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory):
    """A 2-layer random BERT with a character-level WordPiece vocab (no network)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("tiny_bert")
    chars = string.ascii_lowercase + string.digits + ":/._-{}\"(),@*[]=<>!?'"
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(chars) + ["##" + c for c in chars]
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(str(d))
    import torch
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    BertModel(cfg).save_pretrained(str(d))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return str(d)
