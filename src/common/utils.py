from pathlib import Path
from dataclasses import dataclass


# Класс описывающий ограничения на количество генерируемых текстовых фрагментов
@dataclass(frozen=True)
class SamplingBudget:
    chars_per_lang: dict[str, int]
    total_chars: int
    tokenizer_sample_chars: int


def compute_budget(target_tokens: int, overshoot: float, languages: dict) -> SamplingBudget:
    chars = {}
    for lang, cfg in languages.items():
        chars[lang] = int(target_tokens * cfg["share"] * cfg["chars_per_token"] * overshoot)

    total = sum(chars.values())
    sample_chars = min(int(total * 0.01), 1_000_000)

    return SamplingBudget(
        chars_per_lang=chars,
        total_chars=total,
        tokenizer_sample_chars=sample_chars,
    )


def split_budget_by_weights(total: int, weights: list[float]) -> list[int]:
    if not weights:
        return []
    if len(weights) == 1:
        return [total]

    w_sum = sum(weights)
    if w_sum == 0:
        raise ValueError("Сумма весов не может быть равна нулю")
    if any(w < 0 for w in weights):
        raise ValueError("Веса не могут быть отрицательными")

    ideal = [total * w / w_sum for w in weights]
    floors = [int(x) for x in ideal]

    remainder = total - sum(floors)
    if remainder == 0:
        return floors

    fracs = [(ideal[i] - floors[i], i) for i in range(len(weights))]
    fracs.sort(reverse=True)

    for k in range(remainder):
        _, idx = fracs[k]
        floors[idx] += 1

    return floors


# Константы
SHARD_PREFIX = "part_"
SHARD_SUFFIX = ".parquet"

CLEANED_DIR = "cleaned"
TOKENIZER_SAMPLE_DIR = "tokenizer_sample"
TOKENIZER_DIR = "tokenizer"
DATASET_DIR = "dataset"
RUNS_DIR = "runs"
MANIFEST_NAME = "manifest.json"


def shard_filename(shard_index: int) -> str:
    return f"{SHARD_PREFIX}{shard_index:04d}{SHARD_SUFFIX}"


def shard_glob_pattern() -> str:
    return f"{SHARD_PREFIX}*{SHARD_SUFFIX}"


# Пути
def cleaned_dir(base: Path, lang: str, split: str) -> Path:
    """Старый путь (для обратной совместимости, но в новом скрипте не используется)."""
    return base / CLEANED_DIR / lang / split


def cleaned_source_split_dir(base: Path, lang: str, source: str, split: str) -> Path:
    """Новый путь: cleaned/{lang}/{source}/{split}/."""
    return base / CLEANED_DIR / lang / source / split


def cleaned_split_dir(base: Path, split: str) -> Path:
    """Старый общий путь (для обратной совместимости)."""
    return base / CLEANED_DIR / split


def tokenizer_sample_dir(base: Path) -> Path:
    return base / TOKENIZER_SAMPLE_DIR


def tokenizer_dir(base: Path) -> Path:
    return base / TOKENIZER_DIR


def dataset_dir(base: Path, split: str) -> Path:
    return base / DATASET_DIR / split


def run_dir(base: Path, run_id: str) -> Path:
    return base / RUNS_DIR / run_id


def cleaned_manifest_path(base: Path) -> Path:
    return base / CLEANED_DIR / MANIFEST_NAME


def tokenizer_sample_manifest_path(base: Path) -> Path:
    return base / TOKENIZER_SAMPLE_DIR / MANIFEST_NAME


def tokenizer_manifest_path(base: Path) -> Path:
    return base / TOKENIZER_DIR / MANIFEST_NAME


def dataset_manifest_path(base: Path) -> Path:
    return base / DATASET_DIR / MANIFEST_NAME


def run_manifest_path(base: Path, run_id: str) -> Path:
    return base / RUNS_DIR / run_id / MANIFEST_NAME


def dropped_samples_dir(base: Path, lang: str) -> Path:
    return base / CLEANED_DIR / "dropped_samples" / lang