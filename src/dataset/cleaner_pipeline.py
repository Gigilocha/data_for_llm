from typing import Iterable, Iterator
from dataclasses import replace
from .document import Document
from .cleaner import Cleaner
from .rules import split_long


"""
Собирает словарь Cleaner'ов.
    
cleaning_cfg — словарь с правилами (например, {"prose_rules": {...}, "code_rules": {...}}).
languages_cfg — словарь языков (например, {"ru": {"profile": "prose"}, ...}).
"""
def build_cleaners(cleaning_cfg: dict, languages_cfg: dict) -> dict[str, Cleaner]:
    cleaners = {}
    
    for lang, lang_cfg in languages_cfg.items():
        profile_name = lang_cfg["profile"]
        
        # Достаем профиль правил
        if profile_name == "prose":
            profile_cfg = cleaning_cfg["prose_rules"]
        elif profile_name == "code":
            profile_cfg = cleaning_cfg["code_rules"]
        else:
            raise ValueError(f"Unknown profile {profile_name!r} for language {lang!r}")
            
        cleaners[lang] = Cleaner(profile_cfg, language=lang)
        
    return cleaners



"""
Очищает поток документов и режет длинные на чанки.
Возвращает только валидные куски (длина >= min_length).
"""
def clean_stream(
    docs: Iterable[Document], 
    cleaners: dict[str, Cleaner], 
    min_length: int, 
    max_length: int, 
    on_too_long: str
) -> Iterator[Document]:
    for doc in docs:
        # 1. Выбираем Cleaner по языку
        cleaner = cleaners.get(doc.language)
        if cleaner is None:
            # Если языка нет в конфиге — пропускаем (или можно кинуть ошибку)
            continue

        # 2. Очистка
        text = cleaner.clean(doc.text)
        if text is None:
            continue  # Дропнуто правилами (мусор/чужой язык)

        # 3. Нарезка
        chunks = split_long(text, max_length=max_length, on_too_long=on_too_long)

        # 4. Фильтрация и создание новых Document
        for i, chunk in enumerate(chunks):
            if len(chunk) < min_length:
                continue  # Слишком короткий кусок (или исходник, или хвост)

            # Если чанк один и равен исходному тексту (не резали) — parent_id=None
            if len(chunks) == 1:
                yield replace(doc, text=chunk)
            else:
                # Если резали — создаем ID с суффиксом и ставим parent_id
                yield replace(
                    doc, 
                    text=chunk, 
                    id=f"{doc.id}#{i}", 
                    parent_id=doc.id
                )