from typing import Iterator
from dataclasses import replace
import zlib

from .cleaner import Cleaner
from .writer import Writer
from .document import Document
from .loader import load_documents
from .clean_rules import split_long


'''
Заметка: 
    Файл sampler.py - собирает все вместе и считает квоту по каждому источнику.

'''


'''
sample_language - лист дерева. Принимает один поток. Загрузчик, очиститель, два писателя на train и val, отдает количество записанных документов. 
'''
def sample_source(
    source_cfg: dict,
    lang_cfg: dict,
    lang_name: str,
    cleaner: Cleaner,
    budget_chars: int,
    train_writer: Writer,
    val_writer: Writer,
    min_length: int,
    max_length: int,
    on_too_long: str,
    val_split_ratio: float,
    split_salt: str,
) -> int:

    # Количество записанных сиволов во всех документах
    chars_written = 0

    # Создает итератор по источнику, отдает по одному документу
    for doc in load_documents(lang_name=lang_name, lang_cfg=lang_cfg, source_cfg=source_cfg):

        # Если уже записано больше, чем бюджет — выходим
        if chars_written >= budget_chars:
            break

        # Очистка документа
        text = cleaner.clean(doc.text)

        # Проверка на None после очистки
        if text is None:
            continue

        # Контроль длины документа (выброс если меньше мин, разбиение если больше max)
        chunks = split_long(text, max_length=max_length, on_too_long=on_too_long)

        # Проверка каждого чанка при разбивке
        for i, chunk in enumerate(chunks):
            # Если длина чанка меньше мин, отбросить
            if len(chunk) < min_length:
                continue

            # Если чанк всего один перезаписать raw -> cleaned
            if len(chunks) == 1:
                final = replace(doc, text=chunk)

            # Иначе перезаписать с id чанка raw -> cleaned
            else:
                final = replace(doc, text=chunk, id=f"{doc.id}#{i}", parent_id=doc.id)

            # Увеличение счетчика на кол-во символов в тексте текущего докумета
            chars_written += len(chunk)

            # Детерминированный выбор train/val по crc32 (не hash, он рандомизируется между запусками)
            bucket = zlib.crc32(f"{split_salt}{final.parent_id or final.id}".encode("utf-8")) % 100

            # ~1% документов → val
            if bucket < val_split_ratio * 100:
                val_writer.add(final)

            # остальные → train
            else:
                train_writer.add(final)

    # Возвращаем число символов записанных документов
    return chars_written

        
'''
sample_group - 
'''
def sample_group(
    group_name: str,
    budget_chars: int,
    items: list[dict],
    cleaner: Cleaner,
    lang_name: str,
    lang_cfg: str,
    train_writer: Writer,
    val_writer: Writer,
    min_length: int,
    max_length: int,
    on_too_long: str,
    val_split_ratio: float,
    split_salt: str
) -> int:

    # 
