import pytest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from dataset.io.slice_builder import build_tokenizer_sample
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_config():
    """Конфиг для slice_builder (SimpleNamespace вместо MagicMock)."""
    config = SimpleNamespace()
    config.tokenizer = SimpleNamespace(sample_chars=1000)
    config.data = SimpleNamespace(seed=42)
    
    # Язык ru
    lang_ru = SimpleNamespace()
    lang_ru.share = 0.6
    lang_ru.sources = [SimpleNamespace(name="ru_web", weight=1.0)]
    
    # Язык en
    lang_en = SimpleNamespace()
    lang_en.share = 0.4
    lang_en.sources = [SimpleNamespace(name="en_web", weight=1.0)]
    
    config.data.languages = {"ru": lang_ru, "en": lang_en}
    return config

@patch("dataset.io.slice_builder.ds.dataset")
def test_build_sample_respects_budget(mock_dataset, mock_config, tmp_path):
    # Создаем фейковые папки
    for lang in ["ru", "en"]:
        for source in ["ru_web", "en_web"]:
            (tmp_path / "cleaned" / lang / source / "train").mkdir(parents=True, exist_ok=True)

    # Мокаем батч
    mock_batch = MagicMock()
    mock_batch.to_pylist.return_value = [
        {"id": f"doc{i}", "text": "a" * 300, "parent_id": None} 
        for i in range(10)
    ]
    
    mock_dataset.return_value.to_batches.return_value = [mock_batch]
    
    texts = build_tokenizer_sample(mock_config, tmp_path)
    
    total_chars = sum(len(t) for t in texts)
    assert total_chars <= 1000
    assert all(len(t) == 300 for t in texts)


@patch("dataset.io.slice_builder.ds.dataset")
def test_build_sample_truncates_large_doc(mock_dataset, mock_config, tmp_path):
    for lang in ["ru", "en"]:
        for source in ["ru_web", "en_web"]:
            (tmp_path / "cleaned" / lang / source / "train").mkdir(parents=True, exist_ok=True)

    mock_batch = MagicMock()
    mock_batch.to_pylist.return_value = [
        {"id": "huge", "text": "a" * 2000, "parent_id": None}
    ]
    
    mock_dataset.return_value.to_batches.return_value = [mock_batch]
    
    texts = build_tokenizer_sample(mock_config, tmp_path)
    
    assert len(texts) > 0
    assert all(len(t) <= 600 for t in texts)


@patch("dataset.io.slice_builder.ds.dataset")
def test_build_sample_empty_source(mock_dataset, mock_config, tmp_path):
    for lang in ["ru", "en"]:
        for source in ["ru_web", "en_web"]:
            (tmp_path / "cleaned" / lang / source / "train").mkdir(parents=True, exist_ok=True)

    mock_dataset.return_value.to_batches.return_value = []
    
    texts = build_tokenizer_sample(mock_config, tmp_path)
    assert texts == []