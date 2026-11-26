import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import json

from common.logger import get_logger
from common.progress import progress_bar
from common.paths import cleaned_source_split_dir

logger = get_logger(__name__)


def tokenize_split(
    tokenizer: PreTrainedTokenizerFast,
    base_dir: Path,
    config,
    split: str,
    dtype: pa.DataType,
) -> dict:
    """Токенизирует один сплит (train/val) и собирает статистику."""
    stats = {}
    total_tokens = 0
    total_docs = 0

    for lang, lang_cfg in config.data.languages.items():
        for source in lang_cfg.sources:
            src_name = source.name
            in_path = cleaned_source_split_dir(base_dir, lang, src_name, split)
            
            if not in_path.exists():
                logger.warning("Нет папки для %s/%s/%s, пропускаем...", lang, src_name, split)
                continue

            out_dir = base_dir / "tokenized" / lang / src_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "part_0000.parquet"

            ds = load_dataset("parquet", data_files=str(in_path / "*.parquet"), split="train")
            
            src_tokens = 0
            src_docs = 0
            batch_tokens = []
            batch_ids = []
            
            schema = pa.schema([
                ("id", pa.string()),
                ("tokens", pa.list_(dtype)),
            ])
            writer = pq.ParquetWriter(out_file, schema)
            
            try:
                for row in progress_bar(ds, desc=f"[{split}/{lang}/{src_name}]", total=len(ds)):
                    text = row["text"]
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    
                    batch_ids.append(row["id"])
                    batch_tokens.append(ids)
                    
                    src_tokens += len(ids)
                    src_docs += 1
                    
                    if len(batch_tokens) >= 1000:
                        table = pa.Table.from_pydict(
                            {"id": batch_ids, "tokens": batch_tokens}, 
                            schema=schema
                        )
                        writer.write_table(table)
                        batch_tokens = []
                        batch_ids = []
                
                if batch_tokens:
                    table = pa.Table.from_pydict(
                        {"id": batch_ids, "tokens": batch_tokens}, 
                        schema=schema
                    )
                    writer.write_table(table)
            finally:
                writer.close()
            
            avg_len = src_tokens / src_docs if src_docs > 0 else 0
            stats[f"{lang}/{src_name}"] = {
                "tokens": src_tokens,
                "docs": src_docs,
                "avg_len": round(avg_len, 2)
            }
            total_tokens += src_tokens
            total_docs += src_docs
            
            logger.info("[%s/%s/%s] Токенизировано: %d док., %d токенов (avg: %.1f)", 
                        split, lang, src_name, src_docs, src_tokens, avg_len)

    global_avg = total_tokens / total_docs if total_docs > 0 else 0
    return {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "global_avg_len": round(global_avg, 2),
        "sources": stats
    }


def run_tokenization(tokenizer: PreTrainedTokenizerFast, base_dir: Path, config) -> dict:
    """Запускает токенизацию train и val, возвращает полный манифест."""
    vocab_size = tokenizer.vocab_size
    dtype = pa.int16() if vocab_size < 65536 else pa.int32()
    logger.info("Размер словаря: %d. Используем тип токенов: %s", vocab_size, dtype)

    logger.info("=== Токенизация TRAIN ===")
    train_stats = tokenize_split(tokenizer, base_dir, config, "train", dtype)
    
    logger.info("=== Токенизация VAL ===")
    val_stats = tokenize_split(tokenizer, base_dir, config, "val", dtype)

    return {
        "tokenizer_vocab_size": vocab_size,
        "dtype": str(dtype),
        "train": train_stats,
        "val": val_stats,
    }