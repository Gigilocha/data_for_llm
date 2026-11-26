from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir, tokenizer_sample_dir
from dataset.io.slice_builder import build_tokenizer_sample
from tokenizer.tokenizer import train_tokenizer, save_tokenizer
from tokenizer.manifest import build_tokenizer_manifest, write_tokenizer_manifest
from common.progress import progress_bar
from datasets import Dataset
import json

logger = get_logger(__name__)

def main():
    config = get_config()
    env = get_env_settings()
    base_dir = env.output_dir

    logger.info("1. Сбор среза для токенизатора...")
    texts = build_tokenizer_sample(config, base_dir)
    total_chars = sum(len(t) for t in texts)

    sample_dir = tokenizer_sample_dir(base_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample.parquet"
    ds = Dataset.from_dict({"text": texts})
    ds.to_parquet(sample_path)

    logger.info("2. Обучение токенизатора...")
    tokenizer = train_tokenizer(iter(texts), config.tokenizer)
    
    out_dir = tokenizer_dir(base_dir)
    logger.info("3. Сохранение токенизатора...")
    save_tokenizer(tokenizer, out_dir)

    # Манифест
    manifest = build_tokenizer_manifest(
        config=config,
        sample_path=sample_path,
        actual_sample_chars=total_chars,
        output_vocab_size=tokenizer.vocab_size,
    )
    write_tokenizer_manifest(out_dir, manifest)
    logger.info("Готово!")