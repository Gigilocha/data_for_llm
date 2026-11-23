from pathlib import Path
import pyarrow.dataset as ds
import random
from typing import Iterator

from common.logger import get_logger
from common.paths import cleaned_source_split_dir

logger = get_logger(__name__)

def _read_source_docs(base_dir: Path, lang: str, source: str) -> Iterator[dict]:
    """Читает parquet батчами и отдает по одному документу."""
    path = cleaned_source_split_dir(base_dir, lang, source, "train")
    if not path.exists():
        return
        
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            yield row


def build_tokenizer_sample(
    config,
    output_dir: Path,
) -> list[str]:
    """
    Собирает срез для токенизатора.
    Читает cleaned/{lang}/{source}/train/, берет целые документы.
    """
    total_chars = config.tokenizer.sample_chars
    languages_cfg = config.data.languages
    seed = config.data.seed
    rng = random.Random(seed)

    # Бюджет по языкам
    lang_budgets = {
        lang: int(total_chars * cfg.share) 
        for lang, cfg in languages_cfg.items()
    }

    all_texts = []

    for lang, lang_budget in lang_budgets.items():
        sources = languages_cfg[lang].sources
        total_weight = sum(s.weight for s in sources)
        
        logger.info("[%s] start, budget=%d chars", lang, lang_budget)
        written_chars = 0

        for source in sources:
            src_budget = int(lang_budget * (source.weight / total_weight))
            logger.info("  [%s/%s] target=%d chars", lang, source.name, src_budget)

            src_written = 0

            for doc in _read_source_docs(output_dir, lang, source.name):
                text = doc["text"]
                
                # Если документ слишком большой для остатка — пропускаем
                if src_written + len(text) > src_budget:
                    # Но если это первый документ, и он больше бюджета, 
                    # лучше взять его (чтобы не было пусто), или обрезать.
                    # Для токенизатора лучше обрезать, чтобы набрать объем.
                    if src_written == 0:
                        text = text[:src_budget]
                    else:
                        continue

                all_texts.append(text)
                src_written += len(text)
                written_chars += len(text)

                if src_written >= src_budget:
                    break

            logger.info("  [%s/%s] written %d/%d chars", lang, source.name, src_written, src_budget)

        logger.info("[%s] done, total written=%d/%d chars", lang, written_chars, lang_budget)

    rng.shuffle(all_texts)
    return all_texts