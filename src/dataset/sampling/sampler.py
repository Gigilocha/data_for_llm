from dataclasses import replace
import zlib

from ..cleaning.cleaner import Cleaner
from ..io.writer import Writer
from dataset.io.loader import load_documents


def sample_source(
    source_cfg: dict,
    lang_cfg: dict,
    lang_name: str,
    cleaner: Cleaner,
    budget_chars: int,
    train_writer: Writer,
    val_writer: Writer,
    val_split_ratio: float,
    split_salt: str,
    token: str | None = None,
) -> dict[str, dict[str, int]]:

    train_chars = 0
    train_docs = 0
    val_chars = 0
    val_docs = 0

    # Если остаток меньше этого порога — считаем бюджет добитым
    # Берем 0.1% от бюджета, но не меньше 1000 символов
    MIN_REMAINING = max(1000, budget_chars // 1000)
    # Сколько раз подряд мы можем пропустить документ из-за длины
    MAX_MISSES = 500
    misses = 0

    for doc in load_documents(
        lang_name=lang_name, 
        lang_cfg=lang_cfg, 
        source_cfg=source_cfg,
        token=token,
    ):
        used = train_chars + val_chars
        remaining = budget_chars - used

        # 1. Проверка на "добитость" бюджета
        if remaining < MIN_REMAINING:
            break

        # 2. Очистка
        text = cleaner.clean(doc.text)
        if text is None:
            continue

        # 3. Проверка размера документа
        if len(text) > remaining:
            misses += 1
            if misses >= MAX_MISSES:
                break
            continue

        # Документ влез — сбрасываем счетчик промахов
        misses = 0

        # 4. Распределение в train/val
        bucket = zlib.crc32(f"{split_salt}{doc.id}".encode("utf-8")) % 100
        final = replace(doc, text=text)

        if bucket < val_split_ratio * 100:
            val_writer.add(final)
            val_chars += len(text)
            val_docs += 1
        else:
            train_writer.add(final)
            train_chars += len(text)
            train_docs += 1

    return {
        "train": {"chars": train_chars, "docs": train_docs},
        "val":   {"chars": val_chars,   "docs": val_docs},
    }