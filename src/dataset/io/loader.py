from typing import Iterator
import hashlib
from datasets import load_dataset
from common.logger import get_logger
from dataset.document import Document

logger = get_logger(__name__)

def _make_id(source_name: str, key: str) -> str:
    raw = f"{source_name}_{key}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def load_documents(
    lang_name: str, 
    lang_cfg: dict, 
    source_cfg: dict,
    token: str | None = None,   
) -> Iterator[Document]:
    profile = lang_cfg["profile"]
    source_name = source_cfg["name"]

    kwargs = {
        "path": source_cfg["location"],
        "split": source_cfg["split"],
        "revision": source_cfg["revision"],
        "streaming": True,
    }
    if token:
        kwargs["token"] = token
    if source_cfg.get("subset"):
        kwargs["name"] = source_cfg["subset"]

    logger.info("Загрузка датасета: %s", source_name)
    
    try:
        ds = load_dataset(**kwargs)
    except Exception as e:
        logger.error("Ошибка загрузки %s: %s", source_name, e)
        raise

    # Для кода (stack-v3-train) — разворачиваем репозитории в файлы
    if profile == "code":
        lang_filter = source_cfg.get("language_filter", "python")
        try:
            for repo_idx, repo in enumerate(ds):
                for file_idx, f in enumerate(repo.get("files", [])):
                    if (f.get("language") or "").lower() != lang_filter.lower():
                        continue
                    content = f.get("content") or ""
                    if not content or not isinstance(content, str):
                        continue
                    yield Document(
                        id=_make_id(source_name, f"{repo_idx}_{file_idx}"),
                        text=content,
                        source=source_name,
                        language=lang_name,
                        profile=profile,
                        parent_id=None,
                    )
        except Exception as e:
            logger.error("Ошибка во время чтения кода из %s: %s", source_name, e)
            raise
        return

    # Для прозы — одна строка = один документ
    text_field = source_cfg["text_field"]
    try:
        for idx, row in enumerate(ds):
            text = row.get(text_field)
            if not text or not isinstance(text, str):
                continue
            yield Document(
                id=_make_id(source_name, str(idx)),
                text=text,
                source=source_name,
                language=lang_name,
                profile=profile,
                parent_id=None,
            )
    except Exception as e:
        logger.error("Ошибка во время чтения прозы из %s: %s", source_name, e)
        raise