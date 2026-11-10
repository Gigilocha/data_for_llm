1. СТАРТ
   config = get_config()
   cleaners = build_cleaners(cleaning_cfg, languages_cfg)   # {ru: Cleaner, en: ..., python: ...}
   budget = compute_budget(target_tokens, overshoot, languages)

2. ДЛЯ КАЖДОГО ЯЗЫКА (ru, потом en, потом python)
   cleaner = cleaners[lang]
   train_writer = ParquetWriter(f"output/cleaned/{lang}/train/", shard_size)
   val_writer   = ParquetWriter(f"output/cleaned/{lang}/val/",   shard_size)
   chars_written = 0

3. ДЛЯ КАЖДОГО ИСТОЧНИКА ЭТОГО ЯЗЫКА (ru_web, ru_wikipedia)
   raw = load_dataset(..., streaming=True)   # поток dict
   for row in raw:
       if chars_written >= budget[lang]: break        # набрали квоту — хватит

       doc = Document(id=..., text=row["text"], language=lang, ...)
       text = cleaner.clean(doc.text)                 # None = дроп
       if text is None: continue

       for chunk in split_long(text, max_length):
           if len(chunk) < min_length: continue

           final = replace(doc, text=chunk, ...)
           chars_written += len(chunk)

           if bucket(final) < val_ratio: val_writer.add(final)
           else:                         train_writer.add(final)

           if tokenizer_chars < limit:
               tokenizer_writer.add(final)
               tokenizer_chars += len(chunk)

4. КОНЕЦ ЯЗЫКА
   train_writer.flush()
   val_writer.flush()

5. КОНЕЦ ВСЕГО
   tokenizer_writer.flush()
   write manifests