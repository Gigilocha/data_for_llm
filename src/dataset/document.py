from dataclasses import dataclass


'''
Единица корпуса. Проходит через все слои пайплайна:
    loader → clean_stream → tokenizer → writer.

Неизменяема (frozen=True). Правки — только через dataclasses.replace:
    doc2 = replace(doc, text="новый текст")

Поля:
    id         — уникальный идентификатор (стабильный между запусками!)
    text       — сырой или очищенный текст
    source     — откуда пришёл (имя источника из конфига: ru_web, en_web, python)
    language   — язык документа (ru, en, python) — по нему выбирается Cleaner
    profile    — тип данных (prose, code) — определяет набор правил очистки
    parent_id  — для чанков: id исходного документа до split_long.
                 None, если документ не резался. Нужен для распределения
                 по корзинам train/val (корзина считается от parent_id или id,
                 чтобы куски одного документа не разъехались).
'''


# Единица корпуса. Неизменяема: правки — через dataclasses.replace.
@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    source: str
    language: str
    profile: str
    parent_id: str | None = None   # для чанков: id исходного документа

