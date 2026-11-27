# Data for LLM: End-to-End Data Preparation Pipeline

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![uv](https://img.shields.io/badge/package%20manager-uv-black)](https://github.com/astral-sh/uv)

Профессиональный, воспроизводимый и масштабируемый пайплайн для подготовки данных обучения больших языковых моделей (LLM).  
От сырых текстов до оптимизированного бинарного датасета с полным контролем над пропорциями языков, источников и токенов.

## Цель проекта

Получить **полный контроль над данными** для обучения LLM. Пайплайн позволяет из открытых или локальных источников собрать единый очищенный датасет, обучить под него кастомный токенизатор и сформировать итоговый плоский (flat) датасет с гарантированным соблюдением токеновых квот.

Весь процесс воспроизводим, задокументирован и поддерживается манифестами с хэшами.

> **Философия:** При ограниченном размере модели (например, 126M–1B параметров) качество данных важнее их количества. Этот пайплайн реализует принцип «мысль → код» без слепого копирования, с использованием лучших практик индустрии.

---

## Ключевые особенности

- **Идемпотентность** — каждый этап проверяет хэши входных данных и конфигурации. Если ничего не изменилось, этап пропускается.
- **Эффективность памяти** — потоковая обработка с использованием PyArrow и Parquet. Потребление памяти ограничено размером батча.
- **Точный контроль пропорций** — Weighted Round-Robin + ShuffleBuffer гарантируют, что итоговый датасет точно соответствует заданным пропорциям.
- **Строгий контроль бюджета** — сборка останавливается при достижении `target_tokens`.
- **Детальная телеметрия** — на каждом этапе генерируется `manifest.json` со статистикой.
- **Кроссплатформенность** — корректная работа с UTF-8 (включая Windows).

---

## Архитектура пайплайна

Пайплайн состоит из трёх независимых, но связанных этапов:

```mermaid
graph TD
    subgraph S1["Stage 1: RAW → CLEAN"]
        A1["Источники: JSONL / Parquet"] --> A2("Загрузка с лимитами и seed")
        A2 --> A3{"Профиль очистки"}
        A3 -->|"Проза"| A4["Нормализация, эвристики качества"]
        A3 -->|"Код"| A5["Специальные правила для кода"]
        A4 --> A6["Глобальная дедупликация по хэшу"]
        A5 --> A6
        A6 --> A7[("CLEAN: Parquet шарды")]
        A7 -.-> M1["manifest_clean.json"]
    end

    subgraph S2["Stage 2: CLEAN → TOKENIZER"]
        A7 --> B1["Стратифицированный срез данных"]
        B1 --> B2("Обучение BPE токенизатора")
        B2 --> B3{"Оценка: round-trip, fertility, сжатие"}
        B3 --> B4[("Tokenizer: model.json")]
        B4 -.-> M2["manifest_tokenizer.json"]
    end

    subgraph S3["Stage 3: CLEAN + TOKENIZER → FINAL DATASET"]
        A7 --> C1["Потоковая токенизация документов"]
        B4 --> C1
        C1 --> C2{"Weighted Round-Robin + ShuffleBuffer"}
        C2 --> C3["Сборка в flat binary: data.bin + index.parquet"]
        C3 --> C4["Разделение train / val по хэшу ID"]
        C4 --> C5[("FINAL: Готовый датасет")]
        C5 -.-> M3["manifest_dataset.json"]
    end

    M1 -. "Идемпотентность: пропуск при совпадении хэшей" .-> S1
    M2 -. "Идемпотентность" .-> S2
    M3 -. "Идемпотентность" .-> S3

    classDef artifact fill:#d4edda,stroke:#28a745,stroke-width:2px;
    classDef manifest fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 4 2;
    class A7,B4,C5 artifact;
    class M1,M2,M3 manifest;
```

---

# Структура проекта
```
data_for_llm/
├── configs/                # YAML-конфигурации (источники, пропорции, лимиты)
├── scripts/                # Точки входа (CLI)
│   ├── pipeline.py         # Главный оркестратор
│   ├── tokenize_cleaned.py
│   └── build_final_dataset.py
├── src/
│   ├── common/             # Утилиты (config, logger, paths)
│   ├── dataset/            # Ядро работы с данными
│   │   ├── io/
│   │   ├── cleaning/
│   │   └── sampling/
│   └── tokenizer/          # Обучение и валидация токенизатора
├── tests/
├── .env.example
├── pyproject.toml
└── uv.lock
```

---

# Установка
```
git clone https://github.com/Gigilocha/data_for_llm.git
cd data_for_llm

uv sync
cp .env.example .env
```

### Отредактируйте файлы в configs/ под свои источники данных, пропорции и токеновые бюджеты.

---

# Использование

### Полный цикл
```
uv run scripts/pipeline.py --stage all
```

### Отдельные этапы
```
# Этап 1: Очистка и дедупликация
uv run scripts/pipeline.py --stage cleaned

# Этап 2: Обучение токенизатора
uv run scripts/pipeline.py --stage tokenizer

# Этап 3: Токенизация и сборка финального датасета
uv run scripts/pipeline.py --stage final
```

---

# Выходные данные
```
output/
├── cleaned/          # Очищенные тексты (parquet)
├── tokenizer/        # tokenizer.json + manifest.json
├── tokenized/        # Тексты в виде токенов (parquet)
└── dataset/
    ├── train/
    │   ├── data.bin
    │   └── index.parquet
    └── val/
        └── ...
```

---

# Ключевые концепции

| Концепция | Описание |
| --- | --- |
| Бюджет в символах | На этапе очистки набираем тексты в символах (target_tokens × share × chars_per_token × overshoot) |
| Бюджет в токенах | На этапе сборки набираем ровно target_tokens × share токенов |
| Weighted Round-Robin | Берём документ из того языка/источника, который сильнее всего отстаёт от бюджета |
| ShuffleBuffer | Накапливает документы, перемешивает и отдаёт пачками (не разрывая документы) |
| BOS / EOS | Каждый документ оборачивается в <\|bos\|> и <\|eos\|> |

---

# Тестирование
```
# Быстрые unit-тесты
pytest -m unit

# Все тесты
pytest
```

---

# Лицензия

MIT

---

# Автор

Проект развивается в рамках самостоятельного изучения ML/LLM Engineering.Каждый компонент реализован с пониманием «почему именно так» — без слепого копирования.

Telegram: [t.me/Gigilocha](https://t.me/Gigilocha)