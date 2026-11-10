from pathlib import Path
from dataclasses import dataclass


'''
Заметка:
    Файл utils.py содержит вспомогательные функции и классы для работы с корпусом данных.
    Так же здесь определены пути и префиксы файлов
'''


# Класс описывающий ограничения на количество генерируемых текстовых фрагментов
@dataclass(frozen=True)
class SamplingBudget:
    chars_per_lang: dict[str, int]
    total_chars: int
    tokenizer_sample_chars: int

# Счетчик бюджета токенов
def compute_budget(target_tokens: int, overshoot: float, languages: dict) -> SamplingBudget:
    chars = {}
    for lang, cfg in languages.items():
        # target_tokens * share * chars_per_token * overshoot
        chars[lang] = int(target_tokens * cfg["share"] * cfg["chars_per_token"] * overshoot)
    
    # Сэмпл для токенизатора: например, 1% от общего объема, но не больше 1M символов
    total = sum(chars.values())
    sample_chars = min(int(total * 0.01), 1_000_000)
    
    return SamplingBudget(
        chars_per_lang=chars,
        total_chars=total,             
        tokenizer_sample_chars=sample_chars,
    )

# Счётчик бюджета токенов по весам
def split_budget_by_weights(total: int, weights: list[float]) -> list[int]:
    # Обработка пустого списка
    if not weights:
        return []

    # Если весов один — возвращаем весь бюджет одному языку
    if len(weights) == 1:
        return [total]

    # Сумма весов (0.7, 0.2, 0.1) → 1.0
    w_sum = sum(weights) 
    if w_sum == 0:
        raise ValueError("Сумма весов не может быть равна нулю")
    
    if any(w < 0 for w in weights):
        raise ValueError("Веса не могут быть отрицательными")

    # Вычисляем идеальное распределение
    ideal = [total * w / w_sum for w in weights]
    floors = [int(x) for x in ideal]

    remainder = total - sum(floors)
    if remainder == 0:
        return floors

    # индексы, отсортированные по убыванию дробной части
    fracs = [(ideal[i] - floors[i], i) for i in range(len(weights))]
    fracs.sort(reverse=True)

    for k in range(remainder):
        _, idx = fracs[k]
        floors[idx] += 1

    return floors


# Константы (префиксы и имена папок)
SHARD_PREFIX = "part_"
SHARD_SUFFIX = ".parquet"

CLEANED_DIR = "cleaned"
TOKENIZER_SAMPLE_DIR = "tokenizer_sample"
TOKENIZER_DIR = "tokenizer"
DATASET_DIR = "dataset"
RUNS_DIR = "runs"
MANIFEST_NAME = "manifest.json"


# Сборка имен файлов
# Сборка имени шард-файла: префикс + индекс + суффикс
def shard_filename(shard_index: int) -> str:
    return f"{SHARD_PREFIX}{shard_index:04d}{SHARD_SUFFIX}"

# Собираем паттерн для поиска шардов: префикс + * + суффикс
def shard_glob_pattern() -> str:
    return f"{SHARD_PREFIX}*{SHARD_SUFFIX}"


# Сборка путей директорий
# Путь к директории с очищенными данными для конкретного языка и сплита
def cleaned_dir(base: Path, lang: str, split: str) -> Path:
    return base / CLEANED_DIR / lang / split

# Путь к директории с сэмплом для обучения токенизатора
def tokenizer_sample_dir(base: Path) -> Path:
    return base / TOKENIZER_SAMPLE_DIR

# Путь к директории с токенизатором
def tokenizer_dir(base: Path) -> Path:
    return base / TOKENIZER_DIR

# Путь к директории с финальным датасетом 
def dataset_dir(base: Path, split: str) -> Path:
    return base / DATASET_DIR / split

# Путь к директории запусков
def run_dir(base: Path, run_id: str) -> Path:
    return base / RUNS_DIR / run_id


# Сборка путей файлов манифеста
# Путь к файлу манифеста очистки
def cleaned_manifest_path(base: Path) -> Path:
    return base / CLEANED_DIR / MANIFEST_NAME

# Путь к файлу манифеста выборки для токенизатора
def tokenizer_sample_manifest_path(base: Path) -> Path:
    return base / TOKENIZER_SAMPLE_DIR / MANIFEST_NAME

# Путь к файлу манифеста токенизатора
def tokenizer_manifest_path(base: Path) -> Path:
    return base / TOKENIZER_DIR / MANIFEST_NAME

# Путь к файлу манифеста финального датасета
def dataset_manifest_path(base: Path) -> Path:
    return base / DATASET_DIR / MANIFEST_NAME

# Путь к файлу манифеста запусков
def run_manifest_path(base: Path, run_id: str) -> Path:
    return base / RUNS_DIR / run_id / MANIFEST_NAME
