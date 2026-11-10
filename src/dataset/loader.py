from typing import Iterator
import hashlib
from datasets import load_dataset
from .document import Document


"""
Стабильный ID, чтобы при перезапусках корзины не разъезжались.
"""
def _make_id(source_name: str, key: str) -> str:
    raw = f"{source_name}_{key}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


"""
Читает один источник (например, ru_web) и отдает поток Document.

Если у источника поле "text" на верхнем уровне (prose) — оборачивает строку.
Если верхний уровень — репозиторий с files[] (code) — разворачивает в файлы.
"""
def load_documents(lang_name: str, lang_cfg: dict, source_cfg: dict) -> Iterator[Document]:
    profile = lang_cfg["profile"]
    source_name = source_cfg["name"]

    kwargs = {
        "path": source_cfg["location"],
        "split": source_cfg["split"],
        "revision": source_cfg["revision"],
        "streaming": True,
    }
    if source_cfg.get("subset"):
        kwargs["name"] = source_cfg["subset"]

    ds = load_dataset(**kwargs)

    # Для кода (stack-v3-train) — разворачиваем репозитории в файлы
    if profile == "code":
        lang_filter = source_cfg.get("language_filter", "python")
        for repo_idx, repo in enumerate(ds):
            for file_idx, f in enumerate(repo.get("files", [])):
                if f.get("language") != lang_filter:
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
        return

    # Для прозы — одна строка = один документ
    text_field = source_cfg["text_field"]
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