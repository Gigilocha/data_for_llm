from tqdm import tqdm
from typing import Iterable, TypeVar

T = TypeVar("T")

def progress_bar(iterable: Iterable[T], desc: str, total: int | None = None, disable: bool = False) -> Iterable[T]:
    """Обертка над tqdm для красивого вывода."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=disable,
        leave=True,
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )