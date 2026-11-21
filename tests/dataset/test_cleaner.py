import pytest
from dataset.cleaner import Cleaner

@pytest.fixture
def prose_cfg():
    return {
        "spaces": {"enabled": True},
        "strip_control": {"enabled": True},
        "quality": {"enabled": True, "min_alpha_ratio": 0.6}
    }

def test_unknown_rule_raises(prose_cfg):
    prose_cfg["unknown_rule"] = {"enabled": True}
    with pytest.raises(ValueError, match="Unknown rule"):
        Cleaner(prose_cfg, language="ru")

def test_order_is_from_registry_not_config():
    # spaces в YAML раньше strip_control, но в реестре порядок другой
    cfg = {
        "spaces": {"enabled": True},
        "strip_control": {"enabled": True}
    }
    cleaner = Cleaner(cfg, language="ru")
    names = [name for name, _ in cleaner.rules]
    # strip_control (4) должен идти раньше spaces (6) из REGISTRY_ORDER
    assert names.index("strip_control") < names.index("spaces")

def test_clean_applies_rules(prose_cfg):
    cleaner = Cleaner(prose_cfg, language="ru")
    assert cleaner.clean("a    b") == "a b"
    assert cleaner.clean("Привет\u200B мир") == "Привет мир"

def test_clean_drops(prose_cfg):
    cleaner = Cleaner(prose_cfg, language="ru")
    assert cleaner.clean("12345 !!!") is None

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