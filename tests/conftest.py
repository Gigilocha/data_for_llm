import pytest
from pathlib import Path
from unittest.mock import MagicMock
from transformers import PreTrainedTokenizerFast

from common.config import Config, DataConfig, TokenizerConfig, Cleaning
from dataset.cleaning.cleaner import Cleaner


@pytest.fixture
def mock_config() -> Config:
    """Минимальный валидный конфиг для тестов (без чтения YAML)."""
    return Config(
        env=MagicMock(),  # env не нужен в unit-тестах
        data=DataConfig(
            seed=42,
            shard_size_mb=64,
            target_tokens=10000,
            val_split_ratio=0.2,
            split_salt="v1",
            overshoot=1.0,
            context_length=1024,
            languages={
                "ru": {
                    "share": 0.6,
                    "profile": "prose",
                    "chars_per_token": 3.0,
                    "sources": [
                        {
                            "name": "ru_web",
                            "kind": "hf",
                            "location": "org/ds",
                            "split": "train",
                            "text_field": "text",
                            "revision": "abc",
                            "license": "mit",
                            "weight": 1.0,
                        }
                    ],
                },
                "en": {
                    "share": 0.4,
                    "profile": "prose",
                    "chars_per_token": 3.0,
                    "sources": [
                        {
                            "name": "en_web",
                            "kind": "hf",
                            "location": "org/ds2",
                            "split": "train",
                            "text_field": "text",
                            "revision": "def",
                            "license": "mit",
                            "weight": 1.0,
                        }
                    ],
                },
            },
        ),
        cleaning=Cleaning(
            prose_rules={"spaces": {"enabled": True}},
            code_rules={"strip_control": {"enabled": True}},
        ),
        tokenizer=TokenizerConfig(
            vocab_size=2000,
            sample_chars=50000,
            special_tokens={"bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>", "unk": "<|unk|>"},
            split_pattern="",
        ),
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Временная папка для output/ в тестах."""
    return tmp_path / "output"


@pytest.fixture
def fake_tokenizer() -> PreTrainedTokenizerFast:
    """
    Мок токенизатора.
    Не учит реальный BPE, а просто кодирует символы в их unicode-коды.
    """
    tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
    tokenizer.vocab_size = 2000
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    
    # Простая логика: "текст" -> [ord(c) for c in text]
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [ord(c) for c in text]
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "".join(chr(i) for i in ids)
    return tokenizer


@pytest.fixture
def mock_cleaner() -> Cleaner:
    """Cleaner-заглушка, возвращает текст как есть."""
    cleaner = Cleaner({}, language="ru")
    cleaner.clean = lambda text: text
    return cleaner