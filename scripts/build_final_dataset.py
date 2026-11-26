import json
from transformers import PreTrainedTokenizerFast
from common.config import get_config, get_env_settings
from common.logger import get_logger
from common.paths import tokenizer_dir
from dataset.final_builder import build_final_dataset

logger = get_logger(__name__)

def main():
    config = get_config()
    env = get_env_settings()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir(env.output_dir))

    total_target = config.data.target_tokens
    val_tokens = int(total_target * config.data.val_split_ratio)
    train_tokens = total_target - val_tokens

    logger.info("=== Сборка TRAIN (target: %d) ===", train_tokens)
    train_actual, train_dist = build_final_dataset(
        tokenizer, env.output_dir, "train", train_tokens,
        config.data.languages, config.data.context_length, buffer_size=10000
    )

    logger.info("=== Сборка VAL (target: %d) ===", val_tokens)
    val_actual, val_dist = build_final_dataset(
        tokenizer, env.output_dir, "val", val_tokens,
        config.data.languages, config.data.context_length, buffer_size=10000
    )

    manifest = {
        "target_tokens": total_target,
        "context_length": config.data.context_length,
        "train": {"actual_tokens": train_actual, "distribution": train_dist},
        "val": {"actual_tokens": val_actual, "distribution": val_dist},
    }
    manifest_path = env.output_dir / "dataset" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Готово! Манифест записан в %s", manifest_path)