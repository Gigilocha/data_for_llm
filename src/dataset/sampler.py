from typing import Iterator
from dataclasses import replace
import zlib

from common.utils import split_budget_by_weights
from .cleaner import Cleaner
from .writer import Writer
from .document import Document
from .loader import load_documents
from .clean_rules import split_long


'''
Заметка:
    Файл sampler.py — собирает все вместе и считает квоту по каждому источнику.
'''


"""
    Лист дерева: читает один источник, чистит, режет, пишет в train/val.
    Возвращает {"train": {"chars", "docs"}, "val": {"chars", "docs"}}.
"""
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
) -> dict[str, dict[str, int]]:

    train_chars = 0
    train_docs = 0
    val_chars = 0
    val_docs = 0

    for doc in load_documents(lang_name=lang_name, lang_cfg=lang_cfg, source_cfg=source_cfg):

        # Бюджет считается по сумме train + val
        if train_chars + val_chars >= budget_chars:
            break

        text = cleaner.clean(doc.text)
        if text is None:
            continue

        chunks = split_long(text, max_length=max_length, on_too_long=on_too_long)

        for i, chunk in enumerate(chunks):
            if len(chunk) < min_length:
                continue

            if len(chunks) == 1:
                final = replace(doc, text=chunk)
            else:
                final = replace(doc, text=chunk, id=f"{doc.id}#{i}", parent_id=doc.id)

            # Детерминированный выбор train/val по crc32.
            # parent_id or id — чтобы все куски одного документа попали в один split.
            bucket = zlib.crc32(f"{split_salt}{final.parent_id or final.id}".encode("utf-8")) % 100

            if bucket < val_split_ratio * 100:
                val_writer.add(final)
                val_chars += len(chunk)
                val_docs += 1
            else:
                train_writer.add(final)
                train_chars += len(chunk)
                train_docs += 1

    return {
        "train": {"chars": train_chars, "docs": train_docs},
        "val":   {"chars": val_chars,   "docs": val_docs},
    }


"""
    Узел дерева: делит квоту по весам, рекурсивно обходит источники/подгруппы.

    Возвращает:
        written_by_source: {"ru_web": {"train": {"chars", "docs"}, "val": {...}}, ...}
        shortfalls:        [{"group", "source", "written", "budget"}, ...]
        planned_by_source: {"ru_web": 15120000, ...}
    """
def sample_group(
    group_name: str,
    budget_chars: int,
    items: list[dict],
    cleaner: Cleaner,
    lang_name: str,
    lang_cfg: dict,
    train_writer: Writer,
    val_writer: Writer,
    min_length: int,
    max_length: int,
    on_too_long: str,
    val_split_ratio: float,
    split_salt: str,
) -> tuple[
    dict[str, dict[str, dict[str, int]]],   # written_by_source
    list[dict],                              # shortfalls
    dict[str, int],                          # planned_by_source (chars)
]:

    weights = [item["weight"] for item in items]
    budget = split_budget_by_weights(budget_chars, weights)

    written_by_source: dict[str, dict[str, dict[str, int]]] = {}
    planned_by_source: dict[str, int] = {}
    shortfalls: list[dict] = []

    for item, sub_budget in zip(items, budget):
        if "location" in item:
            # ЛИСТ: source
            sub_written = sample_source(
                source_cfg=item,
                lang_cfg=lang_cfg,
                lang_name=lang_name,
                cleaner=cleaner,
                budget_chars=sub_budget,
                train_writer=train_writer,
                val_writer=val_writer,
                min_length=min_length,
                max_length=max_length,
                on_too_long=on_too_long,
                val_split_ratio=val_split_ratio,
                split_salt=split_salt,
            )
            written_by_source[item["name"]] = sub_written
            planned_by_source[item["name"]] = sub_budget

            item_written = sub_written["train"]["chars"] + sub_written["val"]["chars"]

        else:
            # УЗЕЛ: подгруппа
            sub_written, sub_shortfalls, sub_planned = sample_group(
                group_name=f"{group_name}/{item['name']}",
                budget_chars=sub_budget,
                items=item["sources"],
                cleaner=cleaner,
                lang_name=lang_name,
                lang_cfg=lang_cfg,
                train_writer=train_writer,
                val_writer=val_writer,
                min_length=min_length,
                max_length=max_length,
                on_too_long=on_too_long,
                val_split_ratio=val_split_ratio,
                split_salt=split_salt,
            )

            # flatten с префиксом
            for k, v in sub_written.items():
                written_by_source[f"{item['name']}/{k}"] = v
            for k, v in sub_planned.items():
                planned_by_source[f"{item['name']}/{k}"] = v
            shortfalls.extend(sub_shortfalls)

            item_written = sum(
                v["train"]["chars"] + v["val"]["chars"] for v in sub_written.values()
            )

        if item_written < sub_budget:
            shortfalls.append({
                "group": group_name,
                "source": item["name"],
                "written": item_written,
                "budget": sub_budget,
            })

    return written_by_source, shortfalls, planned_by_source