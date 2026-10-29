from functools import partial
from typing import Callable

from dataset.rules import (
    normalize_newlines, fix_encoding, unicode_nfc,
    strip_control, strip_html,
    spacelike, spaces, space_nl, newlines,
    drop_empty_brackets, drop_empty_quotes,
    collapse_repeats, language_id, quality,
)


'''
Реестр правил очистки.

REGISTRY: имя из YAML → фабрика правила.
    Фабрика принимает параметры из YAML (**params) и возвращает
    готовое правило Callable[[str], str | None].

REGISTRY_ORDER: единственный источник правды для порядка вызова.
    YAML задаёт только enabled и параметры, но НЕ порядок.
'''


#
Rule = Callable[[str], str | None]
# Без параметров — фабрика просто возвращает функцию
REGISTRY: dict[str, Callable[..., Rule]] = {
    "normalize_newlines":  lambda: normalize_newlines,
    "fix_encoding":        lambda: fix_encoding,
    "unicode_nfc":         lambda: unicode_nfc,
    "strip_control":       lambda: strip_control,
    "strip_html":          lambda: strip_html,
    "spacelike":           lambda: spacelike,
    "spaces":              lambda: spaces,
    "space_nl":            lambda: space_nl,
    "newlines":            lambda: newlines,
    "drop_empty_brackets": lambda: drop_empty_brackets,
    "drop_empty_quotes":   lambda: drop_empty_quotes,

    # С параметрами — partial
    "collapse_repeats":    lambda max_run=3: partial(collapse_repeats, max_run=max_run),
    "quality":             lambda min_alpha_ratio=0.6: partial(quality, min_alpha_ratio=min_alpha_ratio),

    # language_id: expected_lang подставит Cleaner, остальное — из YAML
    "language_id":         lambda **kw: partial(language_id, **kw),
}


REGISTRY_ORDER: list[str] = [
    "normalize_newlines", # 1. Нормализация переводов строк (первым, иначе \r\n не увидят остальные)
    "fix_encoding", # 2. Кодировка (mojibake) — на байтах, до NFC
    "unicode_nfc", # 3. Unicode NFC
    "strip_control", # 4. Невидимые символы
    "strip_html", # 5. HTML (script/style → comments → tags)
    "spacelike", # 6. Пробелы: NBSP → " ", потом [ \t]+, потом вокруг \n, потом \n{3,}
    "spaces",
    "space_nl",
    "newlines",
    "drop_empty_brackets", # 7. Пустые пары
    "drop_empty_quotes",
    "collapse_repeats", # 8. Повторы знаков
    "language_id", # 9. Фильтры (в конце, после очистки)
    "quality",
]