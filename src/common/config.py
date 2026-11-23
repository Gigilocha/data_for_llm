from pathlib import Path
from typing import Literal
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import PROJECT_ROOT


# Настройки из .env 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
         env_file=".env",
         env_file_encoding="utf-8",
         extra='ignore'
    )

    # Директории
    configs_dir :Path
    output_dir: Path
    hf_token: str | None = None
    fasttext_lang_model: Path | None = None


# Один ресурс
class SourceConfig(BaseModel):
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
class LanguageConfig(BaseModel):
    share: float
    profile: str
    chars_per_token: float
    sources: list[SourceConfig]
    
# Весь конфиг Pre-training данных
class DataConfig(BaseModel):
    seed: int
    shard_size_mb: int
    target_tokens: int
    val_split_ratio: float
    split_salt: str
    overshoot: float
    context_length: int 
    languages: dict[str, LanguageConfig] = Field(default_factory=dict)


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
    prose_rules: ProseRules
    code_rules: CodeRules


# Токенизизатор
class TokenizerConfig(BaseModel):
    vocab_size: int
    sample_chars: int
    special_tokens: dict
    split_pattern: str


# Общий конфиг
class Config(BaseModel):
    env: Settings
    data: DataConfig
    cleaning: Cleaning
    tokenizer: TokenizerConfig


# Загрузка .yaml по названию
def _load_yaml(env: Settings, filename: str) -> dict:
    path = PROJECT_ROOT / env.configs_dir / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Получение настроек
@lru_cache
def get_env_settings() -> Settings:
    return Settings()


# Получение единого конфига
@lru_cache
def get_config() -> Config:
    env = get_env_settings()
    return Config(
        env=env,
        data=DataConfig(**_load_yaml(env, "data_config.yaml")["pretrain"]),
        cleaning=Cleaning(**_load_yaml(env, "cleaning_config.yaml")["pretrain_cleaning"]),
        tokenizer=TokenizerConfig(**_load_yaml(env, "tokenizer_config.yaml")),
    )