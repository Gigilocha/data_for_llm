import sys
import io
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import PreTrainedTokenizerFast

from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir
from dataset.final_builder import build_final_dataset

logger = get_logger(__name__)


def main():
    config = get_config()
    env = get_env_settings()

    # 1. Загружаем токенизатор
    tok_dir = tokenizer_dir(env.output_dir)
    logger.info("Загрузка токенизатора из %s...", tok_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)

    # 2. Считаем бюджеты для train и val
    total_target = config.data.target_tokens
    val_ratio = config.data.val_split_ratio

    val_tokens = int(total_target * val_ratio)
    train_tokens = total_target - val_tokens

    # 3. Запускаем сборку для train
    logger.info("=== Сборка TRAIN (target: %d токенов) ===", train_tokens)
    train_actual, train_dist = build_final_dataset(
        tokenizer=tokenizer,
        base_dir=env.output_dir,
        split="train",  # <--- ДОБАВИЛИ
        target_tokens=train_tokens,
        languages_cfg=config.data.languages,
        context_length=config.data.context_length,
        buffer_size=10000,  # На тесте можно уменьшить, например до 100
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
    )

    # 4. Запускаем сборку для val
    logger.info("=== Сборка VAL (target: %d токенов) ===", val_tokens)
    val_actual, val_dist = build_final_dataset(
        tokenizer=tokenizer,
        base_dir=env.output_dir,
        split="val",    # <--- ДОБАВИЛИ
        target_tokens=val_tokens,
        languages_cfg=config.data.languages,
        context_length=config.data.context_length,
        buffer_size=10000,  # На тесте можно уменьшить
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
    )

    # 5. Собираем манифест
    manifest = {
        "target_tokens": total_target,
        "context_length": config.data.context_length,
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "train": {
            "actual_tokens": train_actual,
            "distribution": train_dist,
            "bin_path": "dataset/train/data.bin",
            "index_path": "dataset/train/index.parquet",
        },
        "val": {
            "actual_tokens": val_actual,
            "distribution": val_dist,
            "bin_path": "dataset/val/data.bin",
            "index_path": "dataset/val/index.parquet",
        },
    }

    manifest_path = env.output_dir / "dataset" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Готово! Манифест записан в %s", manifest_path)


if __name__ == "__main__":
    main()