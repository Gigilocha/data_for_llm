from pathlib import Path


# Определение корня проекта
PROJECT_ROOT = Path(__file__).parents[2].resolve()

# Константы по умолчанию
SHARD_PREFIX = "part_"
SHARD_SUFFIX = ".parquet"

CLEANED_DIR = "cleaned"
TOKENIZER_SAMPLE_DIR = "tokenizer_sample"
TOKENIZER_DIR = "tokenizer"
DATASET_DIR = "dataset"
TOKENIZED_DIR = "tokenized"
RUNS_DIR = "runs"
MANIFEST_NAME = "manifest.json"


def shard_filename(shard_index: int) -> str:
    return f"{SHARD_PREFIX}{shard_index:04d}{SHARD_SUFFIX}"


def shard_glob_pattern() -> str:
    return f"{SHARD_PREFIX}*{SHARD_SUFFIX}"


def cleaned_dir(base: Path, lang: str, split: str) -> Path:
    return base / CLEANED_DIR / lang / split


def cleaned_source_split_dir(base: Path, lang: str, source: str, split: str) -> Path:
    return base / CLEANED_DIR / lang / source / split


def cleaned_split_dir(base: Path, split: str) -> Path:
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


def tokenized_source_split_dir(base: Path, lang: str, source: str, split: str) -> Path:
    return base / TOKENIZED_DIR / lang / source / split