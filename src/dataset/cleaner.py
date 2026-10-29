from dataclasses import dataclass, field
from typing import Callable
from .registry import REGISTRY, REGISTRY_ORDER


'''
Класс Cleaner — сборка и последовательное применение правил очистки.

Создаётся один раз на язык (ru, en, python), потому что language_id
знает ожидаемый язык. Для каждого языка берётся профиль (prose/code),
в котором указаны включённые правила и их параметры.

Сборка (__init__):
    1. Проверяет, что все имена из конфига существуют в REGISTRY
       (опечатка в YAML = ValueError при старте, а не на первом документе).
    2. Идёт по REGISTRY_ORDER (порядок задаёт реестр, не YAML).
    3. Отбирает только правила с enabled=True.
    4. Достаёт параметры (max_run, min_confidence и т.д.).
    5. Подставляет expected_lang для language_id.
    6. Сохраняет пары (имя, готовое правило) и заводит RuleStats.

Очистка (clean):
    - Идёт по цепочке правил.
    - На каждом шаге считает: сколько раз вызвали, сколько раз текст
      изменился (out != text), сколько раз правило дропнуло (None).
    - При первом None останавливается, сохраняет причину и сэмпл
      (первые 20 текстов, обрезанных до 300 символов) и возвращает None.
    - Если все правила пройдены — возвращает очищенный текст.

Статистика (self.stats):
    RuleStats(calls, changed, dropped, samples) на каждое правило.
    По сэмплам удобно настраивать пороги глазами: видно, что именно
    дропается (мусорные цифры, чужой язык, слишком короткий текст).

Не потокобезопасно: счётчики мутируются без Lock. Для v1 — однопоточный
прогон. При переходе на multiprocessing — либо отдельный счётчик на
процесс с последующим reduce, либо вернуть статистику через итератор.

Не знает про Document: работает только со str. Выбор Cleaner по
языку документа — задача clean_stream (шаг 4).
'''


Rule = Callable[[str], str | None]

@dataclass
class RuleStats:
    calls: int = 0
    changed: int = 0
    dropped: int = 0
    samples: list[str] = field(default_factory=list)  # до 20 шт, до 300 символов

class Cleaner:
    def __init__(self, profile_cfg: dict, language: str):
        self.language = language
        self.rules: list[tuple[str, Rule]] = []
        self.stats: dict[str, RuleStats] = {}

        # 1. Проверка: все имена из конфига известны реестру
        for name in profile_cfg:
            if name not in REGISTRY:
                raise ValueError(f"Unknown rule in config: {name!r}")

        # 2. Сборка строго в порядке реестра, а не конфига
        for name in REGISTRY_ORDER:
            if name not in profile_cfg:
                continue
                
            params = profile_cfg[name]
            if not params.get("enabled", False):
                continue

            # Очищаем от служебного флага
            kwargs = {k: v for k, v in params.items() if k != "enabled"}

            # Подставляем expected_lang для language_id
            if name == "language_id":
                kwargs["expected_lang"] = language

            # Создаём правило через фабрику
            rule = REGISTRY[name](**kwargs)
            
            self.rules.append((name, rule))
            self.stats[name] = RuleStats()

    def clean(self, text: str) -> str | None:
        for name, rule in self.rules:
            stats = self.stats[name]
            stats.calls += 1
            
            out = rule(text)
            
            if out is None:
                stats.dropped += 1
                if len(stats.samples) < 20:
                    stats.samples.append(text[:300])
                return None
            
            if out != text:
                stats.changed += 1
            
            text = out
            
        return text