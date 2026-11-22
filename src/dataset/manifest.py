from datetime import datetime, timezone
from pathlib import Path
import json

from common.paths import cleaned_manifest_path, dropped_samples_dir
from dataset.sampling.budget import SamplingBudget


'''
Заметка:
    manifest - это файл с информацией о том как был собран датасет. 
'''


def build_cleaned_manifest(
    *,
    run_id: str,
    target_tokens: int,
    overshoot: float,
    sources: list[dict],
    budget: SamplingBudget,
    chars_per_token: dict[str, float],
    cleaning_rules: dict,
    cleaning_params: dict,
    written_by_lang: dict,
    planned_by_lang: dict,
    shortfalls: list[dict],
    cleaning_stats: dict,
    writers: dict,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "sources": sources,
        "budget": {
            "target_tokens": target_tokens,
            "overshoot": overshoot,
            "total_chars": budget.total_chars,
            "chars_per_lang": budget.chars_per_lang,
            "chars_per_source": planned_by_lang,
            "chars_per_token": chars_per_token,
        },
        "cleaning_rules": cleaning_rules,
        "cleaning_params": cleaning_params,
        "written": written_by_lang,
        "written_totals": writers,  # ← уже готовый dict
        "cleaning_stats": cleaning_stats,
        "shortfalls": shortfalls,
    }


def write_cleaned_manifest(base: Path, manifest: dict) -> None:
    path = cleaned_manifest_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_dropped_samples(base: Path, cleaners: dict) -> None:
    for lang, cleaner in cleaners.items():
        for rule, stats in cleaner.stats.items():
            if not stats.samples:
                continue
            path = dropped_samples_dir(base, lang) / f"{rule}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for text in stats.samples:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")