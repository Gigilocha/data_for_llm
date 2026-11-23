from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict

from common.logger import get_logger

logger = get_logger(__name__)

class BinWriter:
    def __init__(self, bin_path: Path, idx_path: Path, dtype: str = "int16"):
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.dtype = np.dtype(dtype)
        
        # Открываем .bin как memmap (растет по мере записи)
        # Мы будем писать маленькими кусками через append
        self.bin_file = open(self.bin_path, "wb")
        
        self.idx_buffer = []
        self.total_tokens = 0
        self.total_docs = 0
        
        # Схема для idx (parquet)
        self.idx_schema = pa.schema([
            ("doc_id", pa.string()),
            ("source", pa.string()),
            ("language", pa.string()),
            ("start_offset", pa.int64()),
            ("length", pa.int32()),
        ])
        
        logger.info("BinWriter инициализирован. Bin: %s, Idx: %s", bin_path, idx_path)

    def add_document(self, doc_id: str, tokens: List[int], source: str, language: str):
        """Добавляет документ в поток, оборачивая в BOS/EOS."""
        # BOS и EOS должны быть в токенизаторе
        # Для теста возьмем id 1 и 2 (или из конфига)
        # В реальности нужно передавать tokenizer.bos_token_id
        # Но здесь упростим, передадим как аргумент или возьмем глобально.
        
        # Пока пишем без BOS/EOS, чтобы не усложнять. 
        # (Позже добавим параметры bos_id, eos_id)
        
        arr = np.array(tokens, dtype=self.dtype)
        arr.tofile(self.bin_file)
        
        self.idx_buffer.append({
            "doc_id": doc_id,
            "source": source,
            "language": language,
            "start_offset": self.total_tokens,
            "length": len(tokens),
        })
        
        self.total_tokens += len(tokens)
        self.total_docs += 1
        
        # Периодически пишем idx на диск (каждые 10 000 документов)
        if len(self.idx_buffer) >= 10000:
            self._flush_idx()

    def _flush_idx(self):
        """Сбрасывает буфер индекса в parquet."""
        if not self.idx_buffer:
            return
            
        # Если файл еще не создан, создаем с первой записью
        table = pa.Table.from_pylist(self.idx_buffer, schema=self.idx_schema)
        
        if not self.idx_path.exists():
            pq.write_table(table, self.idx_path)
        else:
            # Дописываем в конец (parquet поддерживает append через dataset, но проще перезаписать)
            # Для простоты перезапишем. В реальности лучше писать батчами.
            existing = pq.read_table(self.idx_path)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, self.idx_path)
            
        self.idx_buffer = []

    def close(self):
        """Закрывает файлы."""
        self.bin_file.close()
        self._flush_idx()
        logger.info("BinWriter закрыт. Всего токенов: %d, документов: %d", self.total_tokens, self.total_docs)