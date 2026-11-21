import re
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from common.config import PROJECT_ROOT

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord):
        # Делаем копию записи, чтобы не портить оригинал для файлового хендлера
        record_copy = logging.makeLogRecord(record.__dict__)
        
        levelname = record_copy.levelname
        if levelname in self.COLORS:
            record_copy.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        message = super().format(record_copy)

        if levelname == "INFO":
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        
        return message


def setup_logger(
    name: str = "data_for_llm",
    log_level: str = None,
    log_to_file: bool = True,
    log_file_path: str = "output/logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
   
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_file_str = log_file_path or os.getenv("LOG_FILE_PATH", "output/logs/app.log")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    # 2. Консоль (с цветами)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)

    # 3. Файл (без цветов)
    if log_to_file:
        log_path = Path(log_file_str)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Обычный форматтер, без цветов
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'))
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    return setup_logger(name=name)