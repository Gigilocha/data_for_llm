import pytest
from dataset.cleaning.cleaner import Cleaner
from dataset.cleaning.pipeline import build_cleaners, clean_stream
from dataset.document import Document


pytestmark = pytest.mark.integration  # Это интеграционный тест


@pytest.fixture
def prose_cfg():
    return {
        "spaces": {"enabled": True},
        "strip_control": {"enabled": True},
        "quality": {"enabled": True, "min_alpha_ratio": 0.6},
    }


def test_unknown_rule_raises(prose_cfg):
    prose_cfg["unknown_rule"] = {"enabled": True}
    with pytest.raises(ValueError, match="Unknown rule"):
        Cleaner(prose_cfg, language="ru")


def test_order_is_from_registry_not_config():
    # В YAML "spaces" идет раньше "strip_control", но в REGISTRY_ORDER
    # strip_control (4) должен быть раньше spaces (6)
    cfg = {
        "spaces": {"enabled": True},
        "strip_control": {"enabled": True},
    }
    cleaner = Cleaner(cfg, language="ru")
    names = [name for name, _ in cleaner.rules]
    assert names.index("strip_control") < names.index("spaces")


def test_clean_applies_rules(prose_cfg):
    cleaner = Cleaner(prose_cfg, language="ru")
    assert cleaner.clean("a    b") == "a b"
    assert cleaner.clean("Привет\u200B мир") == "Привет мир"
    assert cleaner.clean("12345 !!!") is None  # quality дропает


def test_stats(prose_cfg):
    cleaner = Cleaner(prose_cfg, language="ru")
    
    cleaner.clean("a    b")      # spaces changed, quality passed
    cleaner.clean("abc")         # no changes, passed
    cleaner.clean("12345 !!!")   # quality dropped
    
    assert cleaner.stats["spaces"].calls == 3
    assert cleaner.stats["spaces"].changed == 1
    assert cleaner.stats["quality"].calls == 3
    assert cleaner.stats["quality"].dropped == 1
    assert cleaner.stats["quality"].samples == ["12345 !!!"]


def test_build_cleaners():
    cleaning_cfg = {
        "prose_rules": {"spaces": {"enabled": True}},
        "code_rules": {"strip_control": {"enabled": True}},
    }
    languages_cfg = {
        "ru": {"profile": "prose"},
        "py": {"profile": "code"},
    }
    cleaners = build_cleaners(cleaning_cfg, languages_cfg)
    assert set(cleaners.keys()) == {"ru", "py"}
    assert "spaces" in [name for name, _ in cleaners["ru"].rules]
    assert "strip_control" in [name for name, _ in cleaners["py"].rules]


def test_clean_stream_drops_none():
    docs = [Document(id="1", text="garbage", source="test", language="ru", profile="prose")]
    
    # Мок Cleaner, который всегда возвращает None
    class MockCleaner:
        language = "ru"
        def clean(self, text):
            return None

    cleaners = {"ru": MockCleaner()}
    out = list(clean_stream(docs, cleaners))
    assert len(out) == 0


def test_clean_stream_preserves_document():
    docs = [Document(id="1", text="hello", source="test", language="ru", profile="prose")]
    
    class MockCleaner:
        language = "ru"
        def clean(self, text):
            return "cleaned text"

    cleaners = {"ru": MockCleaner()}
    out = list(clean_stream(docs, cleaners))
    assert len(out) == 1
    assert out[0].text == "cleaned text"
    assert out[0].id == "1"
    assert out[0].parent_id is None