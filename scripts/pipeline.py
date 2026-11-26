import io
import argparse
from pathlib import Path

from common.config import get_config, get_env_settings
from common.logger import get_logger

# Импорты этапов
from scripts.build_cleaned import main as run_cleaned
from scripts.build_tokenizer import main as run_tokenizer
from scripts.eval_tokenizer import main as run_eval
from scripts.tokenize_cleaned import main as run_tokenize
from scripts.build_final_dataset import main as run_build_dataset


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Data Pipeline")
    parser.add_argument(
        "--stage", 
        choices=[
            "cleaned", 
            "train_tokenizer", 
            "eval_tokenizer", 
            "tokenize_data", 
            "build_dataset", 
            "all"
        ], 
        default="all",
    )
    args = parser.parse_args()

    config = get_config()
    env = get_env_settings()

    logger.info("Запуск пайплайна. Этап: %s", args.stage)

    if args.stage in ["cleaned", "all"]:
        logger.info("--- ЭТАП A: ОЧИСТКА И СБОР ДАННЫХ ---")
        run_cleaned()
    
    if args.stage in ["train_tokenizer", "all"]:
        logger.info("--- ЭТАП B: ОБУЧЕНИЕ ТОКЕНИЗАТОРА ---")
        run_tokenizer()

    if args.stage in ["eval_tokenizer", "all"]:
        logger.info("--- ЭТАП B.1: ОЦЕНКА ТОКЕНИЗАТОРА ---")
        run_eval()

    if args.stage in ["tokenize_data", "all"]:
        logger.info("--- ЭТАП C: ТОКЕНИЗАЦИЯ ВСЕХ ДАННЫХ ---")
        run_tokenize()

    if args.stage in ["build_dataset", "all"]:
        logger.info("--- ЭТАП D: СБОРКА ФИНАЛЬНОГО ДАТАСЕТА ---")
        run_build_dataset()

    logger.info("Пайплайн завершен!")

if __name__ == "__main__":
    main()
