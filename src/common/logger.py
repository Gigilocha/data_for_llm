import re
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from common.config import PROJECT_ROOT  # Оставляем только корень проекта

# Формат логгера
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Синий
        'INFO': '\033[32m',     # Зеленый
        'WARNING': '\033[33m',  # Желтый
        'ERROR': '\033[31m',    # Красный
        'CRITICAL': '\033[35m', # Малиновый
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord):
        levelname = record.levelname
        
        # Красим уровень логирования
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        message = super().format(record)

        # Подсветка чисел и шардов (из проекта Карпатого)
        if levelname == "INFO":
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        
        return message


"""
Настраивает логгер для проекта.
Настройки можно передавать напрямую или задавать через переменные окружения (например, в .env).
"""
def setup_logger(
    name: str = "data_for_llm",
    log_level: str = None,
    log_to_file: bool = True,
    log_file_path: str = "output/logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
   
    # 1. Берем настройки из окружения (если они там есть), иначе используем дефолтные
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_file_str = log_file_path or os.getenv("LOG_FILE_PATH", "output/logs/app.log")
    
    # Конвертируем строку в уровень логирования (INFO -> logging.INFO)
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    
    # Очищаем старые хэндлеры, чтобы при перезапуске (например в Jupyter) не было дублей
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    # 2. Хэндлер для вывода в консоль (с цветами)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console_handler)

    # 3. Хэндлер для записи в файл (без цветов)
    if log_to_file:
        # Формируем полный путь (если путь относительный, привязываем к корню проекта)
        log_path = Path(log_file_str)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
            
        # Создаем папку для логов, если её нет
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # В файл пишем без цветовых кодов и чуть подробнее
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s'))
        logger.addHandler(file_handler)

    return logger

# Удобный хелпер, чтобы не передавать имя каждый раз
def get_logger(name: str) -> logging.Logger:
    return setup_logger(name=name)