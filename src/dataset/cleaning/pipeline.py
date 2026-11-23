from typing import Iterable, Iterator
from dataclasses import replace
from dataset.document import Document
from dataset.cleaning.cleaner import Cleaner


'''
Заметка:
    Файл cleaner_pipeline.py содержит функции для построения пайплайна очистки текста, включая сборку Cleaner'ов и обработку потока документов.
    Функция build_cleaners создает словарь Cleaner'ов на основе конфигурации
'''


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
Очищает поток документов. Возвращает очищенные документы целиком.
"""
def clean_stream(
    docs: Iterable[Document], 
    cleaners: dict[str, Cleaner], 
) -> Iterator[Document]:
    for doc in docs:
        # 1. Выбираем Cleaner по языку
        cleaner = cleaners.get(doc.language)
        if cleaner is None:
            continue

        # 2. Очистка
        text = cleaner.clean(doc.text)
        if text is None:
            continue  # Дропнуто правилами (мусор/чужой язык)

        # 3. Отдаем очищенный документ целиком
        yield replace(doc, text=text)