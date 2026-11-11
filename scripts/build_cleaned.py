import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Импорты проекта 
from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.utils import (
    compute_budget,
    cleaned_split_dir,
)
from dataset.cleaner import Cleaner
from dataset.cleaner_pipeline import build_cleaners       
from dataset.writer import Writer
from dataset.sampler import sample_group
from dataset.manifest import (
    build_cleaned_manifest,
    write_cleaned_manifest,
    write_dropped_samples,
)


"""
Скрипт сборки cleaned/ — Этап A пайплайна.

Читает:
    configs/data_config.yaml, cleaning_config.yaml, tokenizer_config.yaml
    .env (OUTPUT_DIR, HF_TOKEN)

Пишет:
    {OUTPUT_DIR}/cleaned/train/part_XXXX.parquet
    {OUTPUT_DIR}/cleaned/val/part_XXXX.parquet
    {OUTPUT_DIR}/cleaned/manifest.json
    {OUTPUT_DIR}/cleaned/dropped_samples/{lang}/{rule}.jsonl
"""


# Сбор логгера
logger = get_logger(__name__)


# Сбор ID запуска
def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


# Собирает плоский список источников с полем lang.
def collect_sources(languages_cfg: dict) -> list[dict]:
    result = []
    for lang, lang_cfg in languages_cfg.items():
        for src in lang_cfg["sources"]:
            entry = {"lang": lang}
            for key in ("name", "location", "subset", "split", "revision"):
                if key in src and src[key] is not None:
                    entry[key] = src[key]
            # language_filter только для code-профиля
            if "language_filter" in src:
                entry["language_filter"] = src["language_filter"]
            result.append(entry)
    return result


# Из Pydantic-dump оставляет только enabled=True + параметры (без enabled).
def active_rules(rules_cfg: dict) -> dict:
    result = {}
    for name, params in rules_cfg.items():
        if not params.get("enabled", False):
            continue
        result[name] = {k: v for k, v in params.items() if k != "enabled"}
    return result


def main():
    # 1. config + env
    config = get_config()
    env = get_env_settings()

    # 2. HF token → os.environ (loader.py читает os.environ["HF_TOKEN"])
    if config.env.hf_token:
        os.environ.setdefault("HF_TOKEN", config.env.hf_token)

    run_id = make_run_id()
    logger.info("run_id=%s, output=%s", run_id, env.output_dir)

    # 3. данные
    languages_cfg = config.data.languages.model_dump()
    cleaning_cfg = config.cleaning.model_dump(exclude_none=True)

    # 4. cleaners + budget
    cleaners = build_cleaners(cleaning_cfg, languages_cfg)
    budget = compute_budget(
        target_tokens=config.data.target_tokens,
        overshoot=config.data.overshoot,
        languages=languages_cfg,
    )
    logger.info("budget: total=%d chars, per_lang=%s", budget.total_chars, budget.chars_per_lang)

    # 5. writers (train + val), общие на все языки
    shard_bytes = config.data.shard_size_mb * 1024 * 1024
    train_writer = Writer(cleaned_split_dir(env.output_dir, "train"), shard_bytes)
    val_writer   = Writer(cleaned_split_dir(env.output_dir, "val"),   shard_bytes)

    # 6. цикл по языкам
    all_written: dict = {}
    all_planned: dict = {}
    all_shortfalls: list = []

    for lang_name, lang_cfg in languages_cfg.items():
        logger.info("[%s] start, budget=%d chars", lang_name, budget.chars_per_lang[lang_name])

        written, shortfalls, planned = sample_group(
            group_name=lang_name,
            budget_chars=budget.chars_per_lang[lang_name],
            items=lang_cfg["sources"],
            cleaner=cleaners[lang_name],
            lang_name=lang_name,
            lang_cfg=lang_cfg,
            train_writer=train_writer,
            val_writer=val_writer,
            min_length=config.cleaning.min_length,
            max_length=config.cleaning.max_length,
            on_too_long=config.cleaning.on_too_long,
            val_split_ratio=config.data.val_split_ratio,
            split_salt=config.data.split_salt,
        )
        all_written[lang_name] = written
        all_planned[lang_name] = planned
        all_shortfalls.extend(shortfalls)

        for sf in shortfalls:
            logger.warning("[%s] shortfall: %s %d/%d chars",
                           lang_name, sf["source"], sf["written"], sf["budget"])

    # 7. flush
    train_writer.flush()
    val_writer.flush()
    logger.info("flushed: train docs=%d shards=%d, val docs=%d shards=%d",
                train_writer.total_written, train_writer.shard_id,
                val_writer.total_written, val_writer.shard_id)

    # 8. manifest
    manifest = build_cleaned_manifest(
        run_id=run_id,
        target_tokens=config.data.target_tokens,
        overshoot=config.data.overshoot,
        sources=collect_sources(languages_cfg),
        budget=budget,
        chars_per_token={lang: cfg["chars_per_token"] for lang, cfg in languages_cfg.items()},
        cleaning_rules={
            "prose": active_rules(cleaning_cfg["prose_rules"]),
            "code":  active_rules(cleaning_cfg["code_rules"]),
        },
        cleaning_params={
            "min_length": config.cleaning.min_length,
            "max_length": config.cleaning.max_length,
            "on_too_long": config.cleaning.on_too_long,
            "val_split_ratio": config.data.val_split_ratio,
            "split_salt": config.data.split_salt,
        },
        written_by_lang=all_written,
        planned_by_lang=all_planned,
        shortfalls=all_shortfalls,
        cleaning_stats={
            lang: {
                rule: {"calls": s.calls, "changed": s.changed, "dropped": s.dropped}
                for rule, s in cleaner.stats.items()
            }
            for lang, cleaner in cleaners.items()
        },
        writers={"train": train_writer, "val": val_writer},
    )
    write_cleaned_manifest(env.output_dir, manifest)
    write_dropped_samples(env.output_dir, cleaners)

    logger.info("done: manifest=%s", env.output_dir / "cleaned" / "manifest.json")


if __name__ == "__main__":
    main()