import os
from datetime import datetime, timezone

from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import (
    cleaned_source_split_dir,
    cleaned_manifest_path,
)
from dataset.cleaning.pipeline import build_cleaners
from dataset.io.writer import Writer
from dataset.sampling.sampler import sample_source
from dataset.cleaning.manifest import (
    build_cleaned_manifest,
    write_cleaned_manifest,
    write_dropped_samples,
)
from dataset.sampling.budget import compute_budget


# Создание логера
logger = get_logger(__name__)


# Опеределение ID запуска
def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


# Сбор источников данных
def collect_sources(languages_cfg: dict) -> list[dict]:
    result = []
    for lang, lang_cfg in languages_cfg.items():
        for src in lang_cfg["sources"]:
            result.append({"lang": lang, **src})
    return result


# Активные правила очистки
def active_rules(rules_cfg: dict) -> dict:
    result = {}
    for name, params in rules_cfg.items():
        if not params.get("enabled", False):
            continue
        result[name] = {k: v for k, v in params.items() if k != "enabled"}
    return result


# Главная функция
def main():
    # 1. config + env
    config = get_config()
    env = get_env_settings()

    # 2. HF token → os.environ
    if config.env.hf_token:
        os.environ.setdefault("HF_TOKEN", config.env.hf_token)

    run_id = make_run_id()
    logger.info("run_id=%s, output=%s", run_id, env.output_dir)

    # 3. данные
    languages_cfg = config.data.model_dump()["languages"]
    cleaning_cfg = config.cleaning.model_dump(exclude_none=True)

    # 4. cleaners + budget
    cleaners = build_cleaners(cleaning_cfg, languages_cfg)
    budget = compute_budget(
        target_tokens=config.data.target_tokens,
        overshoot=config.data.overshoot,
        languages=languages_cfg,
    )
    logger.info("budget: total=%d chars, per_lang=%s", budget.total_chars, budget.chars_per_lang)

    shard_bytes = config.data.shard_size_mb * 1024 * 1024

    # 5. цикл: lang → source → writers → sample → flush
    all_written: dict = {}
    all_planned: dict = {}
    all_shortfalls: list = []
    writers_summary: dict = {}

    for lang_name, lang_cfg in languages_cfg.items():
        lang_budget = budget.chars_per_lang[lang_name]
        logger.info("[%s] start, budget=%d chars", lang_name, lang_budget)

        for source_cfg in lang_cfg["sources"]:
            source_name = source_cfg["name"]
            weight = source_cfg["weight"]
            src_budget = int(lang_budget * weight)

            train_dir = cleaned_source_split_dir(env.output_dir, lang_name, source_name, "train")
            val_dir = cleaned_source_split_dir(env.output_dir, lang_name, source_name, "val")

            train_writer = Writer(train_dir, shard_bytes)
            val_writer = Writer(val_dir, shard_bytes)

            written = sample_source(
                source_cfg=source_cfg,
                lang_cfg=lang_cfg,
                lang_name=lang_name,
                cleaner=cleaners[lang_name],
                budget_chars=src_budget,
                train_writer=train_writer,
                val_writer=val_writer,
                # min_length, max_length, on_too_long УДАЛЕНЫ
                val_split_ratio=config.data.val_split_ratio,
                split_salt=config.data.split_salt,
                token=env.hf_token,
            )

            train_writer.flush()
            val_writer.flush()

            key = f"{lang_name}/{source_name}"
            all_written[key] = written
            all_planned[key] = src_budget
            writers_summary[f"{key}/train"] = {
                "docs": train_writer.total_written,
                "shards": train_writer.shard_id,
                "bytes": train_writer.total_bytes,
            }
            writers_summary[f"{key}/val"] = {
                "docs": val_writer.total_written,
                "shards": val_writer.shard_id,
                "bytes": val_writer.total_bytes,
            }

            written_total = written["train"]["chars"] + written["val"]["chars"]
            if written_total < src_budget:
                all_shortfalls.append({
                    "lang": lang_name,
                    "source": source_name,
                    "written": written_total,
                    "budget": src_budget,
                })
                logger.warning("[%s/%s] shortfall: %d/%d chars",
                               lang_name, source_name, written_total, src_budget)
            else:
                logger.info("[%s/%s] written %d/%d chars (train=%d, val=%d)",
                            lang_name, source_name, written_total, src_budget,
                            written["train"]["chars"], written["val"]["chars"])

    # 6. manifest
    manifest = build_cleaned_manifest(
        run_id=run_id,
        cleaning_rules={
            "prose": active_rules(cleaning_cfg["prose_rules"]),
            "code":  active_rules(cleaning_cfg["code_rules"]),
        },
        cleaning_params={
            "val_split_ratio": config.data.val_split_ratio,
            "split_salt": config.data.split_salt,
        },
        written_by_lang=all_written,
        planned_by_lang=all_planned,
        shortfalls=all_shortfalls,
        writers=writers_summary,
    )
    write_cleaned_manifest(env.output_dir, manifest)
    write_dropped_samples(env.output_dir, cleaners)

    logger.info("done: manifest=%s", cleaned_manifest_path(env.output_dir))


if __name__ == "__main__":
    main()