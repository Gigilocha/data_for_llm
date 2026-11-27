import logging
import pytest
from common.logger import get_logger


@pytest.mark.unit
def test_logger_creation(tmp_path):
    # Перенаправляем лог в временную папку
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger_module")
    
    # Добавляем FileHandler вручную (так как setup_logger пишет в output/logs)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    logger.addHandler(handler)
    
    logger.info("Тестовое сообщение")
    handler.close()
    
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Тестовое сообщение" in content


@pytest.mark.unit
def test_logger_no_duplicates():
    logger1 = get_logger("same_name")
    logger2 = get_logger("same_name")
    assert logger1 is logger2  # Один и тот же объект