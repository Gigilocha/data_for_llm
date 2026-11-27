# tests/test_sampler.py
import pytest
from pathlib import Path
from dataclasses import replace

from dataset.sampling.sampler import sample_source
from dataset.document import Document
from dataset.io.writer import Writer
from dataset.cleaning.cleaner import Cleaner


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

