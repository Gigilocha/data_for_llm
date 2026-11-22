import sys
import io
import argparse
from pathlib import Path

# Фикс кодировки для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from common.config import get_config, get_env_settings
from common.logger import get_logger

# Импорты этапов
from scripts.build_cleaned import main as run_cleaned
from scripts.build_tokenizer import main as run_tokenizer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM Data Pipeline")
    parser.add_argument(
        "--stage", 
        choices=["cleaned", "tokenizer", "all"], 
        default="all",
        help="Какой этап запускать"
    )
    args = parser.parse_args()

    config = get_config()
    env = get_env_settings()

    logger.info("Запуск пайплайна. Этап: %s", args.stage)

    if args.stage in ["cleaned", "all"]:
        logger.info("--- ЭТАП A: ОЧИСТКА И СБОР ДАННЫХ ---")
        run_cleaned()
    
    if args.stage in ["tokenizer", "all"]:
        logger.info("--- ЭТАП B: СБОР СРЕЗА И ОБУЧЕНИЕ ТОКЕНИЗАТОРА ---")
        run_tokenizer()

    logger.info("Пайплайн завершен!")


if __name__ == "__main__":
    main()