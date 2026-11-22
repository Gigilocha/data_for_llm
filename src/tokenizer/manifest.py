from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib

def build_tokenizer_manifest(
    *,
    config,
    sample_path: Path,
    actual_sample_chars: int,
    output_vocab_size: int,
    evaluation_metrics: dict | None = None,  # <-- новый аргумент
) -> dict:
    """Собирает манифест токенизатора, наследуя данные из конфигов."""
    
    languages = list(config.data.languages.keys())
    
    # Считаем хэш файла среза, чтобы зафиксировать его состояние
    hasher = hashlib.sha256()
    with open(sample_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    sample_sha256 = hasher.hexdigest()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "bpe",
        "library": "tokenizers",
        "type": "multilingual" if len(languages) > 1 else "monolingual",
        "languages": languages,
        "language_prop": {lang: cfg.share for lang, cfg in config.data.languages.items()},
        "vocab_size_requested": config.tokenizer.vocab_size,
        "vocab_size_actual": output_vocab_size,
        "sample_chars_requested": config.tokenizer.sample_chars,
        "sample_chars_actual": actual_sample_chars,
        "sample_sha256": sample_sha256,
        "special_tokens": config.tokenizer.special_tokens,
    }

    # Добавляем метрики, если они есть
    if evaluation_metrics:
        manifest["evaluation"] = evaluation_metrics
        
    return manifest

def write_tokenizer_manifest(save_dir: Path, manifest: dict) -> None:
    path = save_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")