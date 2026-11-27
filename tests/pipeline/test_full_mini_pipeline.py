import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from transformers import PreTrainedTokenizerFast

# Импортируем наши функции из src
from dataset.cleaning.pipeline import build_cleaners
from dataset.io.writer import Writer
from dataset.sampling.sampler import sample_source
from dataset.io.slice_builder import build_tokenizer_sample
from tokenizer.tokenizer import train_tokenizer
from tokenizer.tokenize import run_tokenization
from dataset.final_builder import build_final_dataset

pytestmark = pytest.mark.e2e


@pytest.fixture
def mini_config():
    """Мини-конфиг для E2E теста."""
    config = SimpleNamespace()
    config.data = SimpleNamespace(
        seed=42,
        target_tokens=1000,
        val_split_ratio=0.2,
        split_salt="v1",
        overshoot=1.0,
        context_length=128,
        shard_size_mb=1,
        languages={
            "ru": SimpleNamespace(
                share=0.6,
                profile="prose",
                chars_per_token=3.0,
                sources=[SimpleNamespace(name="ru_web", weight=1.0)]
            ),
            "en": SimpleNamespace(
                share=0.4,
                profile="prose",
                chars_per_token=3.0,
                sources=[SimpleNamespace(name="en_web", weight=1.0)]
            ),
        }
    )
    config.tokenizer = SimpleNamespace(
        vocab_size=300,  # Маленький словарь для теста
        sample_chars=500,
        special_tokens={"bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>", "unk": "<|unk|>"},
        split_pattern=r"\w+",
    )
    config.cleaning = SimpleNamespace(
        prose_rules={"spaces": {"enabled": True}},
        code_rules={"strip_control": {"enabled": True}},
    )
    return config


@patch("dataset.io.loader.load_dataset")
def test_full_pipeline_e2e(mock_load_dataset, mini_config, tmp_path):
    """
    E2E тест: очистка -> токенизатор -> токенизация -> сборка датасета.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # 1. Мокаем загрузку данных (возвращаем 10 фейковых документов)
    def fake_load_dataset(**kwargs):
        print("DEBUG: load_dataset called with", kwargs)
        return [
            {"text": "Это тестовый документ на русском языке. " * 10},
            {"text": "This is a test document in English. " * 10},
        ] * 5  # 10 документов

    mock_load_dataset.side_effect = fake_load_dataset

    # 2. Этап A: Очистка (пишем в cleaned/)
    languages_cfg_dict = {
        lang: {"profile": cfg.profile} 
        for lang, cfg in mini_config.data.languages.items()
    }
    cleaners = build_cleaners(
        cleaning_cfg={
            "prose_rules": mini_config.cleaning.prose_rules, 
            "code_rules": mini_config.cleaning.code_rules
        },
        languages_cfg=languages_cfg_dict,
    )
    
        # 2. Этап A: Очистка (пишем в cleaned/)
    from dataset.document import Document
    
    for lang_name, lang_cfg in mini_config.data.languages.items():
        for source in lang_cfg.sources:
            train_dir = output_dir / "cleaned" / lang_name / source.name / "train"
            train_writer = Writer(train_dir, max_shard_size=10**9)
            
            # Просто пишем по 3 документа вручную
            for i in range(3):
                doc = Document(
                    id=f"{lang_name}_{source.name}_{i}",
                    text=f"Test text for {lang_name} language. " * 10,
                    source=source.name,
                    language=lang_name,
                    profile=lang_cfg.profile,
                )
                train_writer.add(doc)
            
            train_writer.flush()

    # Проверяем, что cleaned/ создан
    assert (output_dir / "cleaned" / "ru" / "ru_web" / "train").exists()
    assert (output_dir / "cleaned" / "en" / "en_web" / "train").exists()

    # 3. Этап B: Токенизатор (срез + обучение)
    sample_texts = build_tokenizer_sample(mini_config, output_dir)
    assert len(sample_texts) > 0

    tokenizer = train_tokenizer(iter(sample_texts), mini_config.tokenizer)
    assert tokenizer.vocab_size <= 300

    # Сохраняем токенизатор
    tok_dir = output_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    # 4. Этап C: Токенизация
    manifest = run_tokenization(tokenizer, output_dir, mini_config)
    assert manifest["train"]["total_tokens"] > 0

    # 5. Этап D: Сборка финального датасета
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    train_actual, train_manifest = build_final_dataset(
        tokenizer=tokenizer,
        base_dir=output_dir,
        split="train",
        target_tokens=1000,
        languages_cfg=mini_config.data.languages,
        context_length=128,
        buffer_size=5,
    )

    # Проверяем, что data.bin и index.parquet созданы
    assert (output_dir / "dataset" / "train" / "data.bin").exists()
    assert (output_dir / "dataset" / "train" / "index.parquet").exists()
    assert train_actual > 0