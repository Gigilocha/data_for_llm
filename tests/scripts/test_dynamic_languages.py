from common.config import Pretrain

def _make_lang(share: float, profile: str = "prose") -> dict:
    return {
        "share": share,
        "profile": profile,
        "chars_per_token": 3.0,
        "sources": [
            {
                "name": "src",
                "kind": "hf",
                "location": "org/ds",
                "split": "train",
                "text_field": "text",
                "revision": "abc",
                "license": "mit",
                "weight": 1.0,
            }
        ],
    }

def test_languages_are_dynamic():
    """Любое количество языков, любые теги — не должно падать."""
    data = {
        "seed": 42,
        "shard_size_mb": 1024,
        "target_tokens": 1000,
        "val_split_ratio": 0.01,
        "split_salt": "v1",
        "overshoot": 2.0,
        "languages": {
            "ru": _make_lang(0.3),
            "en": _make_lang(0.2),
            "py": _make_lang(0.2, "code"),
            "fr": _make_lang(0.1),
            "sql": _make_lang(0.1, "code"),
            "cpp": _make_lang(0.1, "code"),
        },
    }
    cfg = Pretrain(**data)
    assert set(cfg.languages.keys()) == {"ru", "en", "py", "fr", "sql", "cpp"}
    assert cfg.languages["sql"].profile == "code"
    assert cfg.languages["fr"].profile == "prose"

def test_languages_empty_allowed():
    """Пустой словарь языков — валиден (default_factory)."""
    data = {
        "seed": 42, "shard_size_mb": 1024, "target_tokens": 1000,
        "val_split_ratio": 0.01, "split_salt": "v1", "overshoot": 2.0,
        "languages": {},
    }
    cfg = Pretrain(**data)
    assert cfg.languages == {}