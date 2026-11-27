import pytest
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from dataset.io.bin_writer import BinWriter

pytestmark = pytest.mark.unit


def test_bin_writer_writes_data(tmp_path):
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "index.parquet"
    
    writer = BinWriter(bin_path, idx_path, dtype="int16")
    
    writer.add_document("doc1", [1, 2, 3], "ru_web", "ru")
    writer.add_document("doc2", [4, 5], "en_web", "en")
    writer.close()
    
    # Проверяем data.bin
    assert bin_path.exists()
    data = np.fromfile(bin_path, dtype=np.int16)
    assert data.tolist() == [1, 2, 3, 4, 5]
    
    # Проверяем index.parquet
    assert idx_path.exists()
    idx = pq.read_table(idx_path).to_pylist()
    assert len(idx) == 2
    assert idx[0]["doc_id"] == "doc1"
    assert idx[0]["start_offset"] == 0
    assert idx[0]["length"] == 3
    assert idx[1]["doc_id"] == "doc2"
    assert idx[1]["start_offset"] == 3
    assert idx[1]["length"] == 2


def test_bin_writer_empty(tmp_path):
    bin_path = tmp_path / "data.bin"
    idx_path = tmp_path / "index.parquet"
    
    writer = BinWriter(bin_path, idx_path, dtype="int16")
    writer.close()
    
    # bin-файл создан, но пустой
    assert bin_path.exists()
    assert bin_path.stat().st_size == 0
    
    # idx-файл не создан (буфер пуст)
    assert not idx_path.exists()