from pathlib import Path
import pyarrow.dataset as ds
import random
from typing import Iterator
from common.logger import get_logger
from common.paths import cleaned_source_split_dir

logger = get_logger(__name__)

def _read_source_chunks(
    base_dir: Path, 
    lang: str, 
    source: str
) -> Iterator[dict]:
    """Читает parquet батчами и отдает по одному чанку."""
    path = cleaned_source_split_dir(base_dir, lang, source, "train")
    if not path.exists():
        return
        
    dataset = ds.dataset(path, format="parquet")
    # Читаем батчами по 1000 строк (чанков)
    for batch in dataset.to_batches(batch_size=1000):
        for row in batch.to_pylist():
            yield row


def build_tokenizer_sample(
    config,
    output_dir: Path,
) -> list[str]:
    """..."""
    total_chars = config.tokenizer.sample_chars
    languages_cfg = config.data.languages
    seed = config.data.seed
    rng = random.Random(seed)

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
            logger.info("[%s/%s] target=%d chars", lang, source.name, src_budget)

            current_parent = None
            current_doc_chunks = []
            src_written = 0

            for chunk in _read_source_chunks(output_dir, lang, source.name):
                pid = chunk["parent_id"] or chunk["id"]
                
                if current_parent is None:
                    current_parent = pid
                
                if pid != current_parent:
                    doc_text = "".join(current_doc_chunks)
                    if src_written + len(doc_text) <= src_budget:
                        all_texts.append(doc_text)
                        src_written += len(doc_text)
                        written_chars += len(doc_text)
                    
                    current_parent = pid
                    current_doc_chunks = [chunk["text"]]
                else:
                    current_doc_chunks.append(chunk["text"])

                if src_written >= src_budget:
                    break
            
            if current_doc_chunks and src_written < src_budget:
                doc_text = "".join(current_doc_chunks)
                if src_written + len(doc_text) <= src_budget:
                    all_texts.append(doc_text)
                    src_written += len(doc_text)
                    written_chars += len(doc_text)

            logger.info("[%s/%s] written %d/%d chars", lang, source.name, src_written, src_budget)

        logger.info("[%s] done, total written=%d/%d chars", lang, written_chars, lang_budget)

    rng.shuffle(all_texts)
    return all_texts