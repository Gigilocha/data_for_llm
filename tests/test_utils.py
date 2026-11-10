import pytest
from common.utils import SamplingBudget, compute_budget, split_budget_by_weights, shard_filename, shard_glob_pattern, cleaned_dir


def test_compute_budget():
    langs = {
        "ru": {"share": 0.6, "chars_per_token": 3.0},
        "en": {"share": 0.4, "chars_per_token": 3.0},
    }
    b = compute_budget(target_tokens=1_000_000, overshoot=2.0, languages=langs)
    assert b.chars_per_lang["ru"] == 3_600_000
    assert b.chars_per_lang["en"] == 2_400_000
    assert b.total_chars == 6_000_000
    assert b.tokenizer_sample_chars == 60_000   # 1% от 6M

def test_split_budget():
    assert split_budget_by_weights(100, [0.7, 0.3]) == [70, 30]
    assert split_budget_by_weights(100, [0.5, 0.5]) == [50, 50]
    assert split_budget_by_weights(10, [1/3, 1/3, 1/3]) in (
        [3, 3, 4], [3, 4, 3], [4, 3, 3]
    )   # сумма = 10, распределение по остатку
    assert sum(split_budget_by_weights(21_600_000, [0.7, 0.3])) == 21_600_000

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