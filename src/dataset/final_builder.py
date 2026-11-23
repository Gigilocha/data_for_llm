import pyarrow.dataset as ds
from pathlib import Path
from typing import Iterator, Generator
from transformers import PreTrainedTokenizerFast

from common.logger import get_logger
from common.paths import tokenized_source_split_dir
from .io.bin_writer import BinWriter
from .io.shuffle_buffer import ShuffleBuffer

logger = get_logger(__name__)

def _read_tokenized_docs(path: Path, source_name: str, lang: str) -> Generator[dict, None, None]:
    """Читает токенизированные parquet."""
    if not path.exists():
        return

    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            yield {
                "id": row["id"],
                "tokens": row["tokens"],
                "source": source_name,
                "language": lang,
            }


def build_final_dataset(
    tokenizer: PreTrainedTokenizerFast,
    base_dir: Path,
    split: str,
    target_tokens: int,
    languages_cfg: dict,
    context_length: int,
    buffer_size: int = 10000,
    bos_id: int = 1,
    eos_id: int = 2,
) -> Iterator[dict]:
    """
    Собирает финальный датасет в бинарный файл.
    """
    # 1. Создаем писатель
    out_dir = base_dir / "dataset" / split  
    out_dir.mkdir(parents=True, exist_ok=True)
    
    writer = BinWriter(out_dir / "data.bin", out_dir / "index.parquet", dtype="int16")

    # 2. Считаем бюджеты по языкам
    lang_budgets = {lang: int(target_tokens * cfg.share) for lang, cfg in languages_cfg.items()}
    written_tokens = {lang: 0 for lang in lang_budgets}

    # 3. Открываем потоки
    lang_iters = {}
    for lang, lang_cfg in languages_cfg.items():
        for source in lang_cfg.sources:
            path = tokenized_source_split_dir(base_dir, lang, source.name, "train")
            lang_iters[lang] = _read_tokenized_docs(path, source.name, lang)

    active_langs = list(lang_iters.keys())
    buffer = ShuffleBuffer(max_size=buffer_size, seed=42)

    # 4. Weighted Round-Robin
    while active_langs:
        # Выбираем язык с наибольшим отставанием
        lang = min(active_langs, key=lambda l: written_tokens[l] / lang_budgets[l] if lang_budgets[l] > 0 else 1)
        
        try:
            doc = next(lang_iters[lang])
        except StopIteration:
            active_langs.remove(lang)
            continue

        # Оборачиваем в BOS/EOS
        tokens = [bos_id] + doc["tokens"] + [eos_id]
        
        # Добавляем в буфер (он перемешает)
        for buffered_doc in buffer.add({
            "id": doc["id"],
            "tokens": tokens,
            "source": doc["source"],
            "language": lang,
        }):
            # Пишем в бинарный файл
            writer.add_document(
                buffered_doc["id"],
                buffered_doc["tokens"],
                buffered_doc["source"],
                buffered_doc["language"],
            )
            written_tokens[lang] += len(buffered_doc["tokens"])

        if written_tokens[lang] >= lang_budgets[lang]:
            active_langs.remove(lang)

    # 5. Сбрасываем остатки буфера
    for buffered_doc in buffer.flush_all():
        writer.add_document(
            buffered_doc["id"],
            buffered_doc["tokens"],
            buffered_doc["source"],
            buffered_doc["language"],
        )
        written_tokens[lang] += len(buffered_doc["tokens"])

    # 6. Закрываем писатель
    writer.close()
    logger.info("[%s] Финальный датасет собран. Всего токенов: %d", split, writer.total_tokens)
    return writer.total_tokens, dict(written_tokens)