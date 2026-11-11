# tests/unit/test_writer.py
import pyarrow.parquet as pq
from pathlib import Path

from dataset.writer import Writer
from dataset.document import Document


def _doc(id_: str = "1", text: str = "hello") -> Document:
    return Document(
        id=id_,
        text=text,
        source="s",
        language="ru",
        profile="prose",
    )


def test_writes_shard_on_flush(tmp_path):
    writer = Writer(tmp_path, max_shard_size=10**9)
    writer.add(_doc("1", "a" * 100))
    writer.flush()

    files = sorted(tmp_path.glob("shard_*.parquet"))
    assert len(files) == 1
    assert files[0].name == "shard_0000.parquet"

    table = pq.read_table(files[0])
    assert table.num_rows == 1
    assert table.column("id").to_pylist() == ["1"]
    assert table.column("language").to_pylist() == ["ru"]


def test_flush_splits_into_multiple_shards(tmp_path):
    # max_shard_size = 250 байт: документ = 100 + 100 = 200 байт
    # два документа = 400 > 250 → flush перед вторым
    writer = Writer(tmp_path, max_shard_size=250)
    writer.add(_doc("1", "a" * 100))   # buffer=[1], size=200
    writer.add(_doc("2", "b" * 100))   # 200+200>250 → flush([1]) → buffer=[2]
    writer.add(_doc("3", "c" * 100))   # 200+200>250 → flush([2]) → buffer=[3]
    writer.flush()                     # flush([3])

    files = sorted(tmp_path.glob("shard_*.parquet"))
    assert len(files) == 3
    assert writer.shard_id == 3
    assert writer.total_written == 3

    for i, f in enumerate(files):
        table = pq.read_table(f)
        assert table.num_rows == 1
        assert table.column("id").to_pylist() == [str(i + 1)]


def test_empty_flush_is_noop(tmp_path):
    writer = Writer(tmp_path, max_shard_size=10**9)
    writer.flush()   # пустой буфер → ничего

    assert list(tmp_path.glob("shard_*.parquet")) == []
    assert writer.shard_id == 0


def test_total_bytes_accumulates(tmp_path):
    writer = Writer(tmp_path, max_shard_size=10**9)
    writer.add(_doc("1", "a" * 100))   # 100 + 100 = 200
    writer.add(_doc("2", "b" * 200))   # 200 + 100 = 300
    # 200 + 300 = 500
    assert writer.total_bytes == 500
    assert writer.total_written == 2

    writer.flush()
    # total_bytes не обнуляется
    assert writer.total_bytes == 500


def test_flush_resets_current_size_not_total(tmp_path):
    writer = Writer(tmp_path, max_shard_size=1)
    writer.add(_doc("1", "a" * 100))
    writer.flush()                     

    assert writer.current_size == 0
    assert writer.buffer == []
    assert writer.total_bytes == 200
    assert writer.total_written == 1
    assert writer.shard_id == 1


def test_parent_id_and_id_preserved(tmp_path):
    # для чанков: id="abc#0", parent_id="abc"
    writer = Writer(tmp_path, max_shard_size=10**9)
    writer.add(Document(id="abc#0", text="x" * 10, source="s", language="ru",
                        profile="prose", parent_id="abc"))
    writer.add(Document(id="abc#1", text="y" * 10, source="s", language="ru",
                        profile="prose", parent_id="abc"))
    writer.flush()

    files = sorted(tmp_path.glob("shard_*.parquet"))
    table = pq.read_table(files[0])
    assert table.column("id").to_pylist() == ["abc#0", "abc#1"]
    assert table.column("parent_id").to_pylist() == ["abc", "abc"]


def test_multiple_languages_in_one_shard(tmp_path):
    # Writer общий на все языки — в одном шарде могут быть разные language
    writer = Writer(tmp_path, max_shard_size=10**9)
    writer.add(Document(id="1", text="a" * 100, source="s", language="ru", profile="prose"))
    writer.add(Document(id="2", text="b" * 100, source="s", language="en", profile="prose"))
    writer.add(Document(id="3", text="c" * 100, source="s", language="python", profile="code"))
    writer.flush()

    files = sorted(tmp_path.glob("shard_*.parquet"))
    table = pq.read_table(files[0])
    assert table.column("language").to_pylist() == ["ru", "en", "python"]
    assert table.column("profile").to_pylist() == ["prose", "prose", "code"]