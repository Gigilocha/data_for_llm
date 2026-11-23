import sys
import io
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import cleaned_source_split_dir, tokenizer_dir
from dataset.io.writer import Writer  # Используем наш Writer, но он ожидает Document

logger = get_logger(__name__)

def tokenize_split(tokenizer, env, config, split: str, dtype: pa.DataType):
    """Токенизирует один сплит (train/val) и собирает статистику."""
    stats = {}
    total_tokens = 0
    total_docs = 0

    for lang, lang_cfg in config.data.languages.items():
        for source in lang_cfg.sources:
            src_name = source.name
            in_path = cleaned_source_split_dir(env.output_dir, lang, src_name, split)
            
            if not in_path.exists():
                logger.warning("Нет папки для %s/%s/%s, пропускаем...", lang, src_name, split)
                continue

            out_dir = env.output_dir / "tokenized" / lang / src_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "part_0000.parquet"

            # Читаем parquet
            ds = load_dataset("parquet", data_files=str(in_path / "*.parquet"), split="train")
            
            src_tokens = 0
            src_docs = 0
            batch_tokens = []
            batch_ids = []
            
            # Инициализируем ParquetWriter
            schema = pa.schema([
                ("id", pa.string()),
                ("tokens", pa.list_(dtype)),
            ])
            writer = pq.ParquetWriter(out_file, schema)
            
            for row in ds:
                text = row["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                
                batch_ids.append(row["id"])
                batch_tokens.append(ids)
                
                src_tokens += len(ids)
                src_docs += 1
                
                # Пишем батчами по 1000 документов
                if len(batch_tokens) >= 1000:
                    table = pa.Table.from_pydict(
                        {"id": batch_ids, "tokens": batch_tokens}, 
                        schema=schema
                    )
                    writer.write_table(table)
                    batch_tokens = []
                    batch_ids = []
            
            # Дописываем остатки
            if batch_tokens:
                table = pa.Table.from_pydict(
                    {"id": batch_ids, "tokens": batch_tokens}, 
                    schema=schema
                )
                writer.write_table(table)
            
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
    summary = {
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "global_avg_len": round(global_avg, 2),
        "sources": stats
    }
    return summary


def main():
    config = get_config()
    env = get_env_settings()
    
    tok_dir = tokenizer_dir(env.output_dir)
    logger.info("Загрузка токенизатора из %s...", tok_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    
    # Определяем тип данных для токенов
    vocab_size = tokenizer.vocab_size
    dtype = pa.int16() if vocab_size < 65536 else pa.int32()
    logger.info("Размер словаря: %d. Используем тип токенов: %s", vocab_size, dtype)

    logger.info("=== Токенизация TRAIN ===")
    train_stats = tokenize_split(tokenizer, env, config, "train", dtype)
    
    logger.info("=== Токенизация VAL ===")
    val_stats = tokenize_split(tokenizer, env, config, "val", dtype)

    # Собираем манифест
    manifest = {
        "tokenizer_vocab_size": vocab_size,
        "dtype": str(dtype),
        "train": train_stats,
        "val": val_stats,
    }
    
    manifest_path = env.output_dir / "tokenized" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info("Готово! Манифест записан в %s", manifest_path)


if __name__ == "__main__":
    main()