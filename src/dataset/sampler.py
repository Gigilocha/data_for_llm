from dataclasses import replace
import zlib

from .cleaner import Cleaner
from .writer import Writer
from .loader import load_documents
from .clean_rules import split_long


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
    token: str | None = None,
) -> dict[str, dict[str, int]]:

    train_chars = 0
    train_docs = 0
    val_chars = 0
    val_docs = 0

    for doc in load_documents(
        lang_name=lang_name, 
        lang_cfg=lang_cfg, 
        source_cfg=source_cfg,
        token=token,
    ):

        # Ранний выход, если бюджет уже исчерпан (по документу)
        if train_chars + val_chars >= budget_chars:
            break

        text = cleaner.clean(doc.text)
        if text is None:
            continue

        chunks = split_long(text, max_length=max_length, on_too_long=on_too_long)

        for i, chunk in enumerate(chunks):
            if len(chunk) < min_length:
                continue

            # ТА САМАЯ СТАРАЯ ПРОВЕРКА (без + len(chunk))
            if train_chars + val_chars >= budget_chars:
                break

            if len(chunks) == 1:
                final = replace(doc, text=chunk)
            else:
                final = replace(doc, text=chunk, id=f"{doc.id}#{i}", parent_id=doc.id)

            bucket = zlib.crc32(f"{split_salt}{final.parent_id or final.id}".encode("utf-8")) % 100

            if bucket < val_split_ratio * 100:
                val_writer.add(final)
                val_chars += len(chunk)
                val_docs += 1
            else:
                train_writer.add(final)
                train_chars += len(chunk)
                train_docs += 1

        if train_chars + val_chars >= budget_chars:
            break

    return {
        "train": {"chars": train_chars, "docs": train_docs},
        "val":   {"chars": val_chars,   "docs": val_docs},
    }