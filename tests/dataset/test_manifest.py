import json
from types import SimpleNamespace

from dataset.manifest import (
    build_cleaned_manifest,
    write_cleaned_manifest,
    write_dropped_samples,
)
from dataset.sampling.budget import SamplingBudget


def test_build_cleaned_manifest_basic():
    budget = SamplingBudget(
        chars_per_lang={"ru": 1000, "en": 500},
        total_chars=1500,
        tokenizer_sample_chars=15,
    )
    writers = {
        "train": SimpleNamespace(total_written=10, shard_id=2, total_bytes=5000),
        "val":   SimpleNamespace(total_written=1,  shard_id=1, total_bytes=500),
    }

    m = build_cleaned_manifest(
        run_id="test_123",
        target_tokens=100,
        overshoot=2.0,
        sources=[{"lang": "ru", "name": "ru_web", "location": "x"}],
        budget=budget,
        chars_per_token={"ru": 3.0, "en": 3.0},
        cleaning_rules={"prose": {"unicode_nfc": {}}, "code": {}},
        cleaning_params={
            "min_length": 256, "max_length": 8192, "on_too_long": "split",
            "val_split_ratio": 0.01, "split_salt": "v1",
        },
        written_by_lang={
            "ru": {"ru_web": {
                "train": {"chars": 900, "docs": 5},
                "val":   {"chars": 9,   "docs": 1},
            }},
        },
        planned_by_lang={"ru": {"ru_web": 1000}},
        shortfalls=[],
        cleaning_stats={"ru": {"unicode_nfc": {"calls": 10, "changed": 0, "dropped": 0}}},
        writers=writers,
    )

    assert m["run_id"] == "test_123"
    assert m["budget"]["target_tokens"] == 100
    assert m["budget"]["overshoot"] == 2.0
    assert m["budget"]["total_chars"] == 1500
    assert m["budget"]["chars_per_lang"] == {"ru": 1000, "en": 500}
    assert m["budget"]["chars_per_source"] == {"ru": {"ru_web": 1000}}
    assert m["budget"]["chars_per_token"] == {"ru": 3.0, "en": 3.0}

    assert m["written"] == {
        "ru": {"ru_web": {
            "train": {"chars": 900, "docs": 5},
            "val":   {"chars": 9,   "docs": 1},
        }},
    }
    assert m["written_totals"]["train"] == {"docs": 10, "shards": 2, "bytes": 5000}
    assert m["written_totals"]["val"]   == {"docs": 1,  "shards": 1, "bytes": 500}

    assert m["cleaning_rules"] == {"prose": {"unicode_nfc": {}}, "code": {}}
    assert m["cleaning_params"]["min_length"] == 256
    assert m["cleaning_stats"] == {"ru": {"unicode_nfc": {"calls": 10, "changed": 0, "dropped": 0}}}
    assert m["shortfalls"] == []
    assert "created_at" in m


def test_write_cleaned_manifest(tmp_path):
    m = {"run_id": "x", "budget": {}}
    write_cleaned_manifest(tmp_path, m)

    path = tmp_path / "cleaned" / "manifest.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "x"


def test_write_dropped_samples(tmp_path):
    cleaner = SimpleNamespace(stats={
        "quality":     SimpleNamespace(samples=["bad1", "bad2"]),
        "unicode_nfc": SimpleNamespace(samples=[]),   # пусто — файл не создаётся
    })
    cleaners = {"ru": cleaner}

    write_dropped_samples(tmp_path, cleaners)

    path = tmp_path / "cleaned" / "dropped_samples" / "ru" / "quality.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "bad1"
    assert json.loads(lines[1])["text"] == "bad2"

    # пустой samples → файла нет
    assert not (tmp_path / "cleaned" / "dropped_samples" / "ru" / "unicode_nfc.jsonl").exists()