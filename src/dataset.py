'''
Заметка:
    
Данный файл нужен для обработки датасетов, из которых мы хотим получить данные.
    
'''

from pathlib import Path
from typing import Callable, Iterable, Iterator

from datasets import load_dataset
from datasets import interleave_datasets, IterableDataset

# ============================================================================

# Безопасное имя для использования в путях.
def _safe_name(name: str) -> str:
    return name.replace("/", "_")

# Имя папки кеша под конкретный источник.
def _dir_name(source, extra: str | None = None) -> str:
    cache_name = getattr(source, "cache_name", None)
    if cache_name:
        return _safe_name(cache_name)

    name = _safe_name(source.dataset_name)
    subset = getattr(source, "subset", None)
    if subset:
        name = f"{name}__{_safe_name(subset)}"
    if extra:
        name = f"{name}__{_safe_name(extra)}"
    return name

# Путь на диске под конкретный (stage, split, [language], источник)
def _local_dir(data_dir: Path, stage: str, split: str, dirname: str, language: str | None = None) -> Path:
    if language is not None:
        return data_dir / stage / split / language / dirname
    return data_dir / stage / split / dirname

# ============================================================================

# Класс обработки данных
class DataProcessor:
    def __init__(self, data_dir: str, datasets_name: list[str]):
        self.data_dir = data_dir
        self.datasets_name = datasets_name

    # Запуск одного потока данных из сети
    def load_data(self, path: str) -> list[dict] | None:
        try:


