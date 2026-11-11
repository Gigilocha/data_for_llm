from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .document import Document
from dataclasses import asdict


'''
Заметка:
    Файл writer.py накапливает очищенные документы и записывает на диск при превышении порога max_shard_size. Каждый шард — отдельный Parquet-файл.
'''


# Класс Writer отвечает за запись очищенных документов в выходной каталог.
class Writer:
    def __init__(self, directory: Path, max_shard_size: int):
        self.directory = directory # путь к директории
        self.directory.mkdir(parents=True, exist_ok=True) # создаём папку
        self.max_shard_size = max_shard_size # максимальный размер шардов в байтах

        self.buffer = []  # Буффер для накопления документов до flush()
        self.current_size = 0  # Текущий размер буфера в байтах

        self.shard_id = 0 # Номер шарда, используется для именования файлов
        self.total_written = 0 # Общее количество записанных документов
        self.total_bytes = 0   # Общее количество байт, записанных за всё время

    # Буффер накопления документов до max_shard_size. Когда размер буфера превышает shard_size, вызывается flush().
    def add(self, doc: Document) -> None:
        # конвертация документа в словарь для записи в Parquet
        row = asdict(doc)

        # Считаем размер одного документа в байтах (UTF-8) + 100 байт на метаданные
        doc_size = len(row["text"].encode("utf-8")) + 100

        # Если буфер не пустой и добавление превысит лимит — сбрасываем на диск
        if self.buffer and self.current_size + doc_size > self.max_shard_size:
            self.flush()

        self.buffer.append(row)
        self.current_size += doc_size
        self.total_written += 1
        self.total_bytes += doc_size

    # Метод flush() записывает текущий шард на диск.
    def flush(self) -> None:
        # Ничего не пишем, если буфер пустой
        if not self.buffer:
            return

        # Записываем текущий шард. Имя файла: shard_0000.parquet, shard_0001.parquet и т.д.
        pq.write_table(
            pa.Table.from_pylist(self.buffer),
            self.directory / f"shard_{self.shard_id:04d}.parquet",
            compression="zstd",
        )
        self.buffer = []
        self.current_size = 0
        self.shard_id += 1
        # total_bytes НЕ обнуляем — это накопительный счётчик