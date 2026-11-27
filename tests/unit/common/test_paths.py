import pytest
from common.paths import shard_filename, shard_glob_pattern, cleaned_dir


def test_shard_filename():
    assert shard_filename(0) == "part_0000.parquet"
    assert shard_filename(42) == "part_0042.parquet"

def test_shard_glob_matches_filenames():
    pattern = shard_glob_pattern()   # "part_*.parquet"
    # проверка, что имя файла матчится паттерном
    import fnmatch
    assert fnmatch.fnmatch(shard_filename(0), pattern)

def test_cleaned_dir(tmp_path):
    p = cleaned_dir(tmp_path, "ru", "train")
    assert p == tmp_path / "cleaned" / "ru" / "train"
    # и что папка НЕ создана (utils не делает mkdir)
    assert not p.exists()