import pyarrow.dataset as ds
from pathlib import Path
from typing import Generator
from transformers import PreTrainedTokenizerFast

from common.logger import get_logger
from common.paths import tokenized_source_split_dir
from .io.bin_writer import BinWriter
from .io.shuffle_buffer import ShuffleBuffer

logger = get_logger(__name__)

def _read_tokenized_docs(path: Path, source_name: str, lang: str) -> Generator[dict, None, None]:
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
) -> tuple[int, dict]:
    
    out_dir = base_dir / "dataset" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = BinWriter(out_dir / "data.bin", out_dir / "index.parquet", dtype="int16")

    lang_budgets = {lang: int(target_tokens * cfg.share) for lang, cfg in languages_cfg.items()}
    written_tokens = {lang: 0 for lang in lang_budgets}
    
    source_stats = {
        f"{lang}/{src.name}": {"tokens": 0, "docs": 0} 
        for lang, cfg in languages_cfg.items() 
        for src in cfg.sources
    }

    lang_iters = {}
    for lang, lang_cfg in languages_cfg.items():
        for source in lang_cfg.sources:
            path = tokenized_source_split_dir(base_dir, lang, source.name, split)
            lang_iters[lang] = _read_tokenized_docs(path, source.name, lang)

    active_langs = list(lang_iters.keys())
    buffer = ShuffleBuffer(max_size=buffer_size, seed=42)

    try:
        while active_langs:
            lang = min(
                active_langs, 
                key=lambda l: written_tokens[l] / lang_budgets[l] if lang_budgets[l] > 0 else 1
            )
            
            try:
                doc = next(lang_iters[lang])
            except StopIteration:
                active_langs.remove(lang)
                continue

            tokens = [tokenizer.bos_token_id] + doc["tokens"] + [tokenizer.eos_token_id]
            
            # Если добавление документа превысит бюджет, пропускаем
            if written_tokens[lang] + len(tokens) > lang_budgets[lang]:
                continue 
            
            # Кидаем в буфер
            for buffered_doc in buffer.add({
                "id": doc["id"],
                "tokens": tokens,
                "source": doc["source"],
                "language": lang,
            }):
                doc_lang = buffered_doc["language"]
                
                # ПРОВЕРКА БЮДЖЕТА ПРИ ВЫДАЧЕ ИЗ БУФЕРА
                if written_tokens[doc_lang] + len(buffered_doc["tokens"]) > lang_budgets[doc_lang]:
                    continue  # Пропускаем, если уже набрали
                
                writer.add_document(
                    buffered_doc["id"],
                    buffered_doc["tokens"],
                    buffered_doc["source"],
                    buffered_doc["language"],
                )
                
                doc_src = buffered_doc["source"]
                src_key = f"{doc_lang}/{doc_src}"
                
                written_tokens[doc_lang] += len(buffered_doc["tokens"])
                source_stats[src_key]["tokens"] += len(buffered_doc["tokens"])
                source_stats[src_key]["docs"] += 1

        # Сбрасываем остатки буфера
        for buffered_doc in buffer.flush_all():
            doc_lang = buffered_doc["language"]
            
            if written_tokens[doc_lang] + len(buffered_doc["tokens"]) > lang_budgets[doc_lang]:
                continue
            
            writer.add_document(
                buffered_doc["id"],
                buffered_doc["tokens"],
                buffered_doc["source"],
                buffered_doc["language"],
            )
            
            doc_src = buffered_doc["source"]
            src_key = f"{doc_lang}/{doc_src}"
            
            written_tokens[doc_lang] += len(buffered_doc["tokens"])
            source_stats[src_key]["tokens"] += len(buffered_doc["tokens"])
            source_stats[src_key]["docs"] += 1

    finally:
        writer.close()

    logger.info("[%s] Финальный датасет собран. Всего токенов: %d", split, writer.total_tokens)
    
    manifest = {
        "actual_tokens": writer.total_tokens,
        "distribution": source_stats,
    }
    return writer.total_tokens, manifest