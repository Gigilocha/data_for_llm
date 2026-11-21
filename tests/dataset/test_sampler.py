# tests/test_sampler.py
import pytest
from pathlib import Path
from dataclasses import replace

from dataset.sampler import sample_source, sample_group
from dataset.document import Document
from dataset.writer import Writer
from dataset.cleaner import Cleaner


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _doc(id_: str, text: str, lang: str = "ru") -> Document:
    return Document(
        id=id_,
        text=text,
        source="fake",
        language=lang,
        profile="prose",
    )


@pytest.fixture
def noop_cleaner():
    """Cleaner без правил: возвращает текст как есть (или None, если пустой)."""
    return Cleaner({}, language="ru")


# ---------------------------------------------------------------------------
# sample_source
# ---------------------------------------------------------------------------

def test_sample_source_writes_to_train(monkeypatch, tmp_path, noop_cleaner):
    fake_docs = iter([
        _doc("1", "a" * 500),
        _doc("2", "b" * 500),
    ])
    monkeypatch.setattr("dataset.sampler.load_documents", lambda **kw: fake_docs)

    train_writer = Writer(tmp_path / "train", max_shard_size=10**9)
    val_writer = Writer(tmp_path / "val", max_shard_size=10**9)

    written = sample_source(
        source_cfg={"name": "fake", "location": "x"},
        lang_cfg={"profile": "prose", "sources": []},
        lang_name="ru",
        cleaner=noop_cleaner,
        budget_chars=1000,
        train_writer=train_writer,
        val_writer=val_writer,
        min_length=256,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.0,
        split_salt="v1",
    )

    assert written["train"]["chars"] == 1000
    assert written["train"]["docs"] == 2
    assert written["val"]["chars"] == 0
    assert written["val"]["docs"] == 0
    assert train_writer.total_written == 2
    assert val_writer.total_written == 0


def test_sample_source_stops_at_budget(monkeypatch, tmp_path, noop_cleaner):
    fake_docs = iter([_doc(str(i), "a" * 500) for i in range(5)])
    monkeypatch.setattr("dataset.sampler.load_documents", lambda **kw: fake_docs)

    train_writer = Writer(tmp_path / "train", max_shard_size=10**9)
    val_writer = Writer(tmp_path / "val", max_shard_size=10**9)

    written = sample_source(
        source_cfg={"name": "fake", "location": "x"},
        lang_cfg={"profile": "prose", "sources": []},
        lang_name="ru",
        cleaner=noop_cleaner,
        budget_chars=1000,
        train_writer=train_writer,
        val_writer=val_writer,
        min_length=256,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.0,
        split_salt="v1",
    )

    assert written["train"]["chars"] == 1000
    assert written["train"]["docs"] == 2
    assert train_writer.total_written == 2


def test_sample_source_drops_short(monkeypatch, tmp_path, noop_cleaner):
    fake_docs = iter([_doc("1", "a" * 100)])  # 100 < min_length=256
    monkeypatch.setattr("dataset.sampler.load_documents", lambda **kw: fake_docs)

    train_writer = Writer(tmp_path / "train", max_shard_size=10**9)
    val_writer = Writer(tmp_path / "val", max_shard_size=10**9)

    written = sample_source(
        source_cfg={"name": "fake", "location": "x"},
        lang_cfg={"profile": "prose", "sources": []},
        lang_name="ru",
        cleaner=noop_cleaner,
        budget_chars=1000,
        train_writer=train_writer,
        val_writer=val_writer,
        min_length=256,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.0,
        split_salt="v1",
    )

    assert written["train"]["chars"] == 0
    assert written["train"]["docs"] == 0
    assert train_writer.total_written == 0


def test_sample_source_splits_long(monkeypatch, tmp_path, noop_cleaner):
    fake_docs = iter([_doc("abc", "a" * 10000)])
    monkeypatch.setattr("dataset.sampler.load_documents", lambda **kw: fake_docs)

    train_writer = Writer(tmp_path / "train", max_shard_size=10**9)
    val_writer = Writer(tmp_path / "val", max_shard_size=10**9)

    written = sample_source(
        source_cfg={"name": "fake", "location": "x"},
        lang_cfg={"profile": "prose", "sources": []},
        lang_name="ru",
        cleaner=noop_cleaner,
        budget_chars=10**9,
        train_writer=train_writer,
        val_writer=val_writer,
        min_length=256,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.0,
        split_salt="v1",
    )

    assert written["train"]["chars"] == 10000
    assert written["train"]["docs"] == 2   # 8192 + 1808
    assert train_writer.total_written == 2

    ids = [row["id"] for row in train_writer.buffer]
    parents = [row["parent_id"] for row in train_writer.buffer]
    assert ids == ["abc#0", "abc#1"]
    assert parents == ["abc", "abc"]


def test_sample_source_val_bucket(monkeypatch, tmp_path, noop_cleaner):
    # val_split_ratio=1.0 → все в val
    fake_docs = iter([_doc("1", "a" * 500)])
    monkeypatch.setattr("dataset.sampler.load_documents", lambda **kw: fake_docs)

    train_writer = Writer(tmp_path / "train", max_shard_size=10**9)
    val_writer = Writer(tmp_path / "val", max_shard_size=10**9)

    written = sample_source(
        source_cfg={"name": "fake", "location": "x"},
        lang_cfg={"profile": "prose", "sources": []},
        lang_name="ru",
        cleaner=noop_cleaner,
        budget_chars=1000,
        train_writer=train_writer,
        val_writer=val_writer,
        min_length=256,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=1.0,
        split_salt="v1",
    )

    assert written["val"]["chars"] == 500
    assert written["val"]["docs"] == 1
    assert written["train"]["chars"] == 0
    assert written["train"]["docs"] == 0
    assert val_writer.total_written == 1
    assert train_writer.total_written == 0


# ---------------------------------------------------------------------------
# sample_group
# ---------------------------------------------------------------------------

def test_sample_group_splits_budget(monkeypatch):
    calls = []

    def fake_sample_source(source_cfg, budget_chars, **kw):
        calls.append((source_cfg["name"], budget_chars))
        return {
            "train": {"chars": budget_chars, "docs": 1},
            "val":   {"chars": 0, "docs": 0},
        }

    monkeypatch.setattr("dataset.sampler.sample_source", fake_sample_source)

    written, shortfalls, planned = sample_group(
        group_name="ru",
        budget_chars=1000,
        items=[
            {"name": "web", "location": "x", "weight": 0.7},
            {"name": "wiki", "location": "y", "weight": 0.3},
        ],
        cleaner=None,
        lang_name="ru",
        lang_cfg={"profile": "prose", "sources": []},
        train_writer=None,
        val_writer=None,
        min_length=1,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.01,
        split_salt="v1",
    )

    assert shortfalls == []
    assert planned == {"web": 700, "wiki": 300}
    assert written == {
        "web":  {"train": {"chars": 700, "docs": 1}, "val": {"chars": 0, "docs": 0}},
        "wiki": {"train": {"chars": 300, "docs": 1}, "val": {"chars": 0, "docs": 0}},
    }
    assert calls == [("web", 700), ("wiki", 300)]


def test_sample_group_shortfall(monkeypatch):
    def fake_sample_source(source_cfg, budget_chars, **kw):
        return {
            "train": {"chars": budget_chars // 2, "docs": 1},
            "val":   {"chars": 0, "docs": 0},
        }

    monkeypatch.setattr("dataset.sampler.sample_source", fake_sample_source)

    written, shortfalls, planned = sample_group(
        group_name="ru",
        budget_chars=1000,
        items=[{"name": "web", "location": "x", "weight": 1.0}],
        cleaner=None,
        lang_name="ru",
        lang_cfg={"profile": "prose", "sources": []},
        train_writer=None,
        val_writer=None,
        min_length=1,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.01,
        split_salt="v1",
    )

    assert planned == {"web": 1000}
    assert shortfalls == [{
        "group": "ru",
        "source": "web",
        "written": 500,
        "budget": 1000,
    }]


def test_sample_group_recurses(monkeypatch):
    calls = []

    def fake_sample_source(source_cfg, budget_chars, **kw):
        calls.append((source_cfg["name"], budget_chars))
        return {
            "train": {"chars": budget_chars, "docs": 1},
            "val":   {"chars": 0, "docs": 0},
        }

    monkeypatch.setattr("dataset.sampler.sample_source", fake_sample_source)

    written, shortfalls, planned = sample_group(
        group_name="ru",
        budget_chars=1000,
        items=[{
            "name": "web_group",
            "weight": 1.0,
            "sources": [
                {"name": "web1", "location": "x", "weight": 0.5},
                {"name": "web2", "location": "y", "weight": 0.5},
            ],
        }],
        cleaner=None,
        lang_name="ru",
        lang_cfg={"profile": "prose", "sources": []},
        train_writer=None,
        val_writer=None,
        min_length=1,
        max_length=8192,
        on_too_long="split",
        val_split_ratio=0.01,
        split_salt="v1",
    )

    assert shortfalls == []
    # flatten с префиксом
    assert planned == {"web_group/web1": 500, "web_group/web2": 500}
    assert written == {
        "web_group/web1": {"train": {"chars": 500, "docs": 1}, "val": {"chars": 0, "docs": 0}},
        "web_group/web2": {"train": {"chars": 500, "docs": 1}, "val": {"chars": 0, "docs": 0}},
    }
    assert calls == [("web1", 500), ("web2", 500)]