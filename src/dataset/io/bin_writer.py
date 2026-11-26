from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from common.logger import get_logger


logger = get_logger(__name__)


class BinWriter:
    def __init__(self, bin_path: Path, idx_path: Path, dtype: str = "int16"):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.dtype = np.dtype(dtype)
        
        try:
            self.bin_file = open(self.bin_path, "wb")
        except OSError as e:
            logger.error("Не удалось открыть файл %s для записи: %s", self.bin_path, e)
            raise
            
        self.idx_buffer = []
        self.total_tokens = 0
        self.total_docs = 0
        
        self.idx_schema = pa.schema([
            ("doc_id", pa.string()),
            ("source", pa.string()),
            ("language", pa.string()),
            ("start_offset", pa.int64()),
            ("length", pa.int32()),
        ])
        
        logger.info("BinWriter инициализирован: %s", bin_path)

    def add_document(self, doc_id: str, tokens: List[int], source: str, language: str):
        arr = np.array(tokens, dtype=self.dtype)
        try:
            arr.tofile(self.bin_file)
        except OSError as e:
            logger.error("Ошибка записи в %s: %s", self.bin_path, e)
            raise
            
        self.idx_buffer.append({
            "doc_id": doc_id,
            "source": source,
            "language": language,
            "start_offset": self.total_tokens,
            "length": len(tokens),
        })
        
        self.total_tokens += len(tokens)
        self.total_docs += 1
        
        if len(self.idx_buffer) >= 10000:
            self._flush_idx()

    def _flush_idx(self):
        if not self.idx_buffer:
            return
            
        table = pa.Table.from_pylist(self.idx_buffer, schema=self.idx_schema)
        
        try:
            if not self.idx_path.exists():
                pq.write_table(table, self.idx_path)
            else:
                existing = pq.read_table(self.idx_path)
                combined = pa.concat_tables([existing, table])
                pq.write_table(combined, self.idx_path)
        except Exception as e:
            logger.error("Ошибка записи индекса %s: %s", self.idx_path, e)
            raise
            
        self.idx_buffer = []

    def close(self):
        self.bin_file.close()
        self._flush_idx()
        logger.info("BinWriter закрыт. Всего токенов: %d, документов: %d", self.total_tokens, self.total_docs)