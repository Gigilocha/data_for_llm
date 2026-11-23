import random
from typing import Iterator, Any, List
from collections import Counter


class ShuffleBuffer:
    """
    Буфер для перемешивания документов.
    Накапливает документы, при заполнении перемешивает и отдает половину.
    """
    def __init__(self, max_size: int, seed: int = 42):
        self.max_size = max_size
        self.buffer: List[Any] = []
        self.rng = random.Random(seed)
        self.stats = Counter()

    def add(self, item: Any) -> Iterator[Any]:
        """Добавляет элемент. Если буфер полон — отдает перемешанную пачку."""
        # Считаем статистику по языкам для манифеста
        lang = item.get("language", "unknown")
        self.stats[lang] += 1
        
        self.buffer.append(item)
        
        # Когда буфер заполнен, сбрасываем половину
        if len(self.buffer) >= self.max_size:
            yield from self._flush()

    def _flush(self) -> Iterator[Any]:
        """Перемешивает буфер и отдает первую половину."""
        self.rng.shuffle(self.buffer)
        
        # Отдаем половину, вторую оставляем для следующего микса
        half = len(self.buffer) // 2
        to_yield = self.buffer[:half]
        self.buffer = self.buffer[half:]
        
        yield from to_yield

    def flush_all(self) -> Iterator[Any]:
        """Отдает всё, что осталось в буфере (в конце пайплайна)."""
        if self.buffer:
            self.rng.shuffle(self.buffer)
            yield from self.buffer
            self.buffer = []