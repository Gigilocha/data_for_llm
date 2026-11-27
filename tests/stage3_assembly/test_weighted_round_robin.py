import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from dataset.final_builder import build_final_dataset

pytestmark = pytest.mark.integration


@patch("dataset.final_builder.BinWriter")
@patch("dataset.final_builder.tokenized_source_split_dir")
@patch("dataset.final_builder._read_tokenized_docs")
def test_round_robin_distribution(
    mock_read_docs, mock_split_dir, mock_bin_writer, tmp_path
):
    tokenizer = MagicMock()
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    
    mock_split_dir.return_value = tmp_path
    
    def read_docs_side_effect(path, source_name, lang):
        if lang == "ru":
            for i in range(10):
                yield {"id": f"ru_{i}", "tokens": [10] * 100, "source": "ru_web", "language": "ru"}
        else:
            for i in range(10):
                yield {"id": f"en_{i}", "tokens": [20] * 100, "source": "en_web", "language": "en"}
    
    mock_read_docs.side_effect = read_docs_side_effect
    
    languages_cfg = {
        "ru": SimpleNamespace(
            share=0.6, 
            sources=[SimpleNamespace(name="ru_web", weight=1.0)]
        ),
        "en": SimpleNamespace(
            share=0.4, 
            sources=[SimpleNamespace(name="en_web", weight=1.0)]
        ),
    }
    
    # НАСТРАИВАЕМ МОК BinWriter
    mock_writer_instance = MagicMock()
    mock_writer_instance.total_tokens = 1000
    mock_bin_writer.return_value = mock_writer_instance
    
    actual_tokens, distribution = build_final_dataset(
        tokenizer=tokenizer,
        base_dir=tmp_path,
        split="train",
        target_tokens=1000,
        languages_cfg=languages_cfg,
        context_length=1024,
        buffer_size=5,
    )
    
    assert actual_tokens == 1000  # Теперь это число
    assert distribution["distribution"]["ru/ru_web"]["tokens"] > 0
    assert distribution["distribution"]["en/en_web"]["tokens"] > 0  