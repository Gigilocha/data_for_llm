import pytest
from pydantic import ValidationError
from common.config import DataConfig, LanguageConfig, SourceConfig, TokenizerConfig


@pytest.mark.unit
def test_source_config_valid():
    data = {
        "name": "ru_web",
        "kind": "hf",
        "location": "HuggingFaceFW/fineweb-2",
        "subset": "rus_Cyrl",
        "split": "train",
        "text_field": "text",
        "revision": "abc123",
        "license": "odc-by",
        "weight": 1.0,
    }
    cfg = SourceConfig(**data)
    assert cfg.name == "ru_web"
    assert cfg.weight == 1.0


@pytest.mark.unit
def test_source_config_missing_field():
    data = {"name": "ru_web", "kind": "hf"}  # не хватает location, split и т.д.
    with pytest.raises(ValidationError):
        SourceConfig(**data)


@pytest.mark.unit
def test_data_config_dynamic_languages():
    # Проверяем, что можно передать любое количество языков
    data = {
        "seed": 42,
        "shard_size_mb": 64,
        "target_tokens": 1000,
        "val_split_ratio": 0.2,
        "split_salt": "v1",
        "overshoot": 1.0,
        "context_length": 1024,
        "languages": {
            "fr": {
                "share": 1.0,
                "profile": "prose",
                "chars_per_token": 3.0,
                "sources": [
                    {
                        "name": "fr_web",
                        "kind": "hf",
                        "location": "org/ds",
                        "split": "train",
                        "text_field": "text",
                        "revision": "abc",
                        "license": "mit",
                        "weight": 1.0,
                    }
                ],
            }
        },
    }
    cfg = DataConfig(**data)
    assert "fr" in cfg.languages
    assert cfg.languages["fr"].share == 1.0


@pytest.mark.unit
def test_tokenizer_config_valid():
    data = {
        "vocab_size": 16000,
        "sample_chars": 1000000,
        "special_tokens": {"bos": "<|bos|>", "eos": "<|eos|>"},
        "split_pattern": r"\w+",
    }
    cfg = TokenizerConfig(**data)
    assert cfg.vocab_size == 16000