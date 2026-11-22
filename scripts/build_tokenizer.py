import sys
import io
from pathlib import Path

# Фикс кодировки для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from datasets import Dataset

from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir, tokenizer_sample_dir
from dataset.io.slice_builder import build_tokenizer_sample
from tokenizer.tokenizer import train_tokenizer, save_tokenizer
from tokenizer.manifest import build_tokenizer_manifest, write_tokenizer_manifest

logger = get_logger(__name__)

def main():
    config = get_config()
    env = get_env_settings()
    base_dir = env.output_dir

    logger.info("1. Сбор среза для токенизатора")
    texts = build_tokenizer_sample(config, base_dir)
    
    if not texts:
        raise ValueError("Срез пустой! Проверь, есть ли данные в cleaned/")

    total_chars = sum(len(t) for t in texts)
    logger.info("Собрано %d документов, %d символов.", len(texts), total_chars)

    sample_dir = tokenizer_sample_dir(base_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample.parquet"
    
    logger.info("2. Сохранение среза в %s...", sample_path)
    ds = Dataset.from_dict({"text": texts})
    ds.to_parquet(sample_path)
    logger.info("   Срез сохранен.")

    logger.info("3. Обучение токенизатора...")
    tokenizer = train_tokenizer(iter(texts), config.tokenizer)
    
    actual_vocab_size = tokenizer.vocab_size
    logger.info("Токенизатор обучен. Реальный размер словаря: %d", actual_vocab_size)

    out_dir = tokenizer_dir(base_dir)
    logger.info("4. Сохранение токенизатора в %s...", out_dir)
    save_tokenizer(tokenizer, out_dir)

    logger.info("5. Оценка качества токенизатора...")
    from tokenizer.evaluate import evaluate_tokenizer
    eval_metrics = evaluate_tokenizer(tokenizer, texts)
    logger.info("Compression Ratio: %s, Tokens per Word: %s", 
                eval_metrics["compression_ratio"], eval_metrics["tokens_per_word"])

    logger.info("6. Запись манифеста...")
    manifest = build_tokenizer_manifest(
        config=config,
        sample_path=sample_path,
        actual_sample_chars=total_chars,
        output_vocab_size=actual_vocab_size,
        evaluation_metrics=eval_metrics,  # <-- передаем сюда
    )
    write_tokenizer_manifest(out_dir, manifest)
    
    logger.info("Готово! Токенизатор обучен и сохранен.")

if __name__ == "__main__":
    main()