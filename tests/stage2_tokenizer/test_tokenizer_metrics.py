import pytest
from tokenizer.evaluate import evaluate_tokenizer

pytestmark = pytest.mark.unit


def test_evaluate_tokenizer_basic(fake_tokenizer):
    # fake_tokenizer.encode("abc") -> [97, 98, 99] (3 токена)
    # Длина текста 3, токенов 3 -> compression_ratio = 1.0
    texts = ["abc", "def"]
    metrics = evaluate_tokenizer(fake_tokenizer, texts)
    
    assert metrics["total_texts_evaluated"] == 2
    assert metrics["compression_ratio"] == 1.0
    assert metrics["tokens_per_word"] == 3.0
    assert metrics["roundtrip_success_rate"] == 1.0


def test_evaluate_tokenizer_empty(fake_tokenizer):
    metrics = evaluate_tokenizer(fake_tokenizer, [])
    assert "error" in metrics


def test_evaluate_tokenizer_roundtrip(fake_tokenizer):
    # Проверяем, что decode(encode(x)) == x
    text = "Привет мир"
    ids = fake_tokenizer.encode(text, add_special_tokens=False)
    decoded = fake_tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == text