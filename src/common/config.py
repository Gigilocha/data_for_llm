from pathlib import Path
from typing import Literal
from functools import lru_cache

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------------------------------

# Определение корня проекта
PROJECT_ROOT = Path(__file__).parents[2].resolve()

# ---------------------------------------------------------------------------------------------------

# Настройки из .env 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
         env_file=".env",
         env_file_encoding="utf-8",
         extra='ignore'
    )

    # Директории
    configs_dir :Path
    data_dir: Path
    output_dir: Path

# ---------------------------------------------------------------------------------------------------

# Один ресурс
class Sources(BaseModel):
    name: str
    kind: str
    location: str
    subset: str | None = None
    split: str
    text_field: str
    revision: str
    license: str
    weight: float

# Один язык
class Language(BaseModel):
    share: float
    profile: str
    chars_per_token: float
    sources: list[Sources]

# Все языки
class AllLanguages(BaseModel):
    ru: Language
    en: Language
    python: Language

# Весь конфиг Pre-training данных
class Pretrain(BaseModel):
    seed: int
    shard_size_mb: int
    target_tokens: int
    val_split_ratio: float
    split_salt: str
    overshoot: float
    languages: AllLanguages

# ---------------------------------------------------------------------------------------------------

class RuleConfig(BaseModel):
    enabled: bool = True
    model_config = {"extra": "allow"} # Дополнительные параметры конкретных правил (max_run, min_confidence и т.д.)

class ProseRules(BaseModel):
    normalize_newlines: RuleConfig | None = None
    fix_encoding: RuleConfig | None = None
    unicode_nfc: RuleConfig | None = None
    strip_control: RuleConfig | None = None
    strip_html: RuleConfig | None = None
    spacelike: RuleConfig | None = None
    spaces: RuleConfig | None = None
    space_nl: RuleConfig | None = None
    newlines: RuleConfig | None = None
    drop_empty_brackets: RuleConfig | None = None
    drop_empty_quotes: RuleConfig | None = None
    collapse_repeats: RuleConfig | None = None
    language_id: RuleConfig | None = None
    quality: RuleConfig | None = None

class CodeRules(BaseModel):
    normalize_newlines: RuleConfig | None = None
    unicode_nfc: RuleConfig | None = None
    strip_control: RuleConfig | None = None

class Cleaning(BaseModel):
    max_length: int
    min_length: int
    on_too_long: Literal["split", "drop"]
    prose_rules: ProseRules
    code_rules: CodeRules

# ---------------------------------------------------------------------------------------------------

# Токенизизатор
class Tokenizer(BaseModel):
    algorithm: str
    library: str
    vocab_size: int
    train_sample_size: int | None = None
    special_tokens: dict
    split_pattern: str

# ---------------------------------------------------------------------------------------------------

# Общий конфиг
class Config(BaseModel):
    env: Settings
    data: Pretrain
    cleaning: Cleaning
    tokenizer: Tokenizer

# ---------------------------------------------------------------------------------------------------

# Загрузка .yaml по названию
def _load_yaml(env: Settings, filename: str) -> dict:
    path = PROJECT_ROOT / env.configs_dir / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ---------------------------------------------------------------------------------------------------

# Получение настроек
@lru_cache
def get_env_settings() -> Settings:
    return Settings()

# ---------------------------------------------------------------------------------------------------

# Получение единого конфига
@lru_cache
def get_config() -> Config:
    env = get_env_settings()
    return Config(
        env=env,
        data=Pretrain(**_load_yaml(env, "data_config.yaml")["pretrain"]),
        cleaning=Cleaning(**_load_yaml(env, "cleaning_config.yaml")["pretrain_cleaning"]),
        tokenizer=Tokenizer(**_load_yaml(env, "tokenizer_config.yaml")),
    )