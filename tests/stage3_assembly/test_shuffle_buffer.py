import pytest
from dataset.io.shuffle_buffer import ShuffleBuffer

pytestmark = pytest.mark.unit


def test_buffer_accumulates_and_shuffles():
    buffer = ShuffleBuffer(max_size=10, seed=42)
    
    # Добавляем 10 документов
    docs = [{"id": str(i), "language": "ru"} for i in range(10)]
    output = []
    
    for doc in docs:
        output.extend(buffer.add(doc))
    
    # Буфер заполнился (10 >= 10), отдал половину (5) и перемешал
    assert len(output) == 5
    assert len(buffer.buffer) == 5
    
    # Проверяем, что отданные документы перемешаны (не по порядку 0,1,2,3,4)
    assert [d["id"] for d in output] != ["0", "1", "2", "3", "4"]


def test_buffer_flush_all():
    buffer = ShuffleBuffer(max_size=10, seed=42)
    
    for i in range(3):
        list(buffer.add({"id": str(i), "language": "ru"}))
    
    # В буфере 3 документа (меньше max_size=10), ничего не отдано
    assert len(buffer.buffer) == 3
    
    # Принудительно сбрасываем всё
    output = list(buffer.flush_all())
    assert len(output) == 3
    assert len(buffer.buffer) == 0


def test_buffer_stats():
    buffer = ShuffleBuffer(max_size=10, seed=42)
    
    list(buffer.add({"id": "1", "language": "ru"}))
    list(buffer.add({"id": "2", "language": "ru"}))
    list(buffer.add({"id": "3", "language": "en"}))
    
    assert buffer.stats["ru"] == 2
    assert buffer.stats["en"] == 1