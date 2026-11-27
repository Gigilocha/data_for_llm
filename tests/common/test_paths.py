import pytest
from pathlib import Path
from common.paths import (
    shard_filename,
    shard_glob_pattern,
    cleaned_source_split_dir,
    tokenized_source_split_dir,
    tokenizer_dir,
    dataset_dir,
)


@pytest.mark.unit
def test_shard_filename():
    assert shard_filename(0) == "part_0000.parquet"
    assert shard_filename(42) == "part_0042.parquet"


@pytest.mark.unit
def test_shard_glob_pattern():
    import fnmatch
    pattern = shard_glob_pattern()
    assert fnmatch.fnmatch(shard_filename(0), pattern)
    assert fnmatch.fnmatch(shard_filename(999), pattern)


@pytest.mark.unit
def test_cleaned_source_split_dir(tmp_path):
    p = cleaned_source_split_dir(tmp_path, "ru", "ru_web", "train")
    assert p == tmp_path / "cleaned" / "ru" / "ru_web" / "train"


@pytest.mark.unit
def test_tokenized_source_split_dir(tmp_path):
    p = tokenized_source_split_dir(tmp_path, "en", "en_web", "val")
    assert p == tmp_path / "tokenized" / "en" / "en_web" / "val"


@pytest.mark.unit
def test_tokenizer_dir(tmp_path):
    p = tokenizer_dir(tmp_path)
    assert p == tmp_path / "tokenizer"


@pytest.mark.unit
def test_dataset_dir(tmp_path):
    p = dataset_dir(tmp_path, "train")
    assert p == tmp_path / "dataset" / "train"