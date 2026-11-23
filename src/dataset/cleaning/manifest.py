from datetime import datetime, timezone
from pathlib import Path
import json

from common.paths import cleaned_manifest_path, dropped_samples_dir


def build_cleaned_manifest(
    *,
    run_id: str,
    cleaning_rules: dict,
    cleaning_params: dict,
    written_by_lang: dict,
    planned_by_lang: dict,
    shortfalls: list[dict],
    writers: dict,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "cleaning_rules": cleaning_rules,
        "cleaning_params": cleaning_params,
        "planned_chars_by_source": planned_by_lang,
        "written_chars": written_by_lang,
        "written_totals": writers,
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