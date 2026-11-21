import pytest
from dataset.cleaner_pipeline import build_cleaners, clean_stream
from dataset.cleaner import Cleaner
from dataset.document import Document

@pytest.fixture
def dummy_cfg():
    return {
        "cleaning": {
            "prose_rules": {"spaces": {"enabled": True}},
            "code_rules": {"strip_control": {"enabled": True}}
        },
        "languages": {
            "ru": {"profile": "prose"},
            "python": {"profile": "code"}
        }
    }

def test_build_cleaners(dummy_cfg):
    cleaners = build_cleaners(dummy_cfg["cleaning"], dummy_cfg["languages"])
    
    assert set(cleaners.keys()) == {"ru", "python"}
    assert "spaces" in [name for name, _ in cleaners["ru"].rules]
    assert "strip_control" in [name for name, _ in cleaners["python"].rules]


class MockCleaner:
    """Простой мок Cleaner для тестов пайплайна, чтобы не зависеть от правил."""
    def __init__(self, text_to_return: str | None):
        self.text_to_return = text_to_return
        self.language = "mock"

    def clean(self, text: str) -> str | None:
        return self.text_to_return


def test_clean_stream_basic():
    docs = [Document(id="1", text="hello", source="test", language="ru", profile="prose")]
    cleaners = {"ru": MockCleaner("cleaned text")}
    
    out = list(clean_stream(docs, cleaners, min_length=1, max_length=100, on_too_long="split"))
    assert len(out) == 1
    assert out[0].text == "cleaned text"
    assert out[0].id == "1"
    assert out[0].parent_id is None


def test_clean_stream_drops_none():
    docs = [Document(id="1", text="garbage", source="test", language="ru", profile="prose")]
    cleaners = {"ru": MockCleaner(None)}
    
    out = list(clean_stream(docs, cleaners, min_length=1, max_length=100, on_too_long="split"))
    assert len(out) == 0


def test_clean_stream_min_length():
    docs = [Document(id="1", text="text", source="test", language="ru", profile="prose")]
    cleaners = {"ru": MockCleaner("short")}
    
    # "short" = 5 символов, min_length=10
    out = list(clean_stream(docs, cleaners, min_length=10, max_length=100, on_too_long="split"))
    assert len(out) == 0


def test_clean_stream_split_long():
    docs = [Document(id="1", text="long", source="test", language="ru", profile="prose")]
    cleaners = {"ru": MockCleaner("a" * 20)}
    
    # max_length=10, min_length=5
    out = list(clean_stream(docs, cleaners, min_length=5, max_length=10, on_too_long="split"))
    
    assert len(out) == 2
    assert out[0].id == "1#0"
    assert out[0].text == "a" * 10
    assert out[0].parent_id == "1"
    
    assert out[1].id == "1#1"
    assert out[1].text == "a" * 10
    assert out[1].parent_id == "1"


def test_clean_stream_drop_long():
    docs = [Document(id="1", text="long", source="test", language="ru", profile="prose")]
    cleaners = {"ru": MockCleaner("a" * 20)}
    
    # on_too_long="drop", поэтому 20 символов при max_length=10 должны отвалиться
    out = list(clean_stream(docs, cleaners, min_length=1, max_length=10, on_too_long="drop"))
    assert len(out) == 0