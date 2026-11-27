import pytest
from dataset.cleaning.rules import (
    drop_empty_brackets,
    drop_empty_quotes,
    collapse_repeats,
    quality,
    language_id,
)

pytestmark = pytest.mark.unit


# --- drop_empty_brackets ---
@pytest.mark.parametrize("src, expected", [
    ("текст () ещё", "текст  ещё"),
    ("текст [ ] ещё", "текст  ещё"),
    ("текст {  } ещё", "текст  ещё"),
    ("()", None),
    ("( ) [ ] { }", None),
    # Не удаляются (прилипли к слову)
    ("print()", "print()"),
    ("x = f()", "x = f()"),
    # Не пустые
    ("(a)", "(a)"),
    ("( a )", "( a )"),
    ("обычный текст", "обычный текст"),
], ids=[
    "round", "square", "curly", "only-round", "only-all",
    "print", "f-call",
    "round-nonempty", "spaced-nonempty", "noop",
])
def test_drop_empty_brackets(src, expected):
    assert drop_empty_brackets(src) == expected


# --- drop_empty_quotes ---
@pytest.mark.parametrize("src, expected", [
    ("Он сказал «» и всё", "Он сказал  и всё"),
    ("«»", None),
    ("«  »", None),
    ("„ “", None),
    ("“ ”", None),
    ("‘ ’", None),
    ("‹ ›", None),
    # Прилипли к слову
    ("a«»b", "ab"),
    # Не пустые
    ("«ok»", "«ok»"),
    ("«  x  »", "«  x  »"),
    ("обычный текст", "обычный текст"),
], ids=[
    "ru-empty", "only-ru", "ru-spaces", "de-empty", "en-empty",
    "single-empty", "angle-empty",
    "glued",
    "ru-nonempty", "spaced-nonempty", "noop",
])
def test_drop_empty_quotes(src, expected):
    assert drop_empty_quotes(src) == expected


# --- collapse_repeats (max_run=3) ---
@pytest.mark.parametrize("src, expected", [
    ("Ну!!!!", "Ну!!!"),
    ("Ну!!!!!", "Ну!!!"),
    ("Да--------", "Да---"),
    ("x====y", "x===y"),
    ("a....b", "a...b"),
    ("a!!!!!b?????c", "a!!!b???c"),
    # Не трогаются (ровно 3 или меньше)
    ("Ну!!!", "Ну!!!"),
    ("a..b", "a..b"),
    ("a.b", "a.b"),
    ("a!?!", "a!?!"),
], ids=[
    "4-to-3", "5-to-3", "dashes", "eq-4-to-3", "dots-4-to-3", "mixed-runs",
    "3-noop", "2-noop", "1-noop", "different-signs",
])
def test_collapse_repeats(src, expected):
    assert collapse_repeats(src, max_run=3) == expected


# --- quality (min_alpha_ratio=0.6) ---
@pytest.mark.parametrize("src, expected", [
    ("Привет мир", "Привет мир"),
    ("Hello, world!", "Hello, world!"),
    ("abc", "abc"),
    ("a b c", "a b c"),
    # Дропаются (букв < 60%)
    ("abc 123", None),
    ("a1b2c3", None),
    ("12345 !!!", None),
    ("!!!???", None),
    ("12345", None),
    ("", None),
    ("   ", None),
], ids=[
    "ru", "en", "letters-only", "spaced-letters",
    "half-letters", "alternating", "digits-punct", "punct-only",
    "digits-only", "empty", "spaces",
])
def test_quality(src, expected):
    assert quality(src) == expected


# --- language_id (с моком detect) ---
@pytest.mark.parametrize("src, expected_lang, mock_detected, mock_conf, expected", [
    # Уверенно на русском, ожидаем русский -> пропускаем
    ("Москва — столица России", "ru", "ru", 0.95, "Москва — столица России"),
    # Уверенно на английском, ожидаем русский -> дроп
    ("Moscow is the capital", "ru", "en", 0.95, None),
    # Низкая уверенность -> дроп
    ("какой-то текст", "ru", "ru", 0.3, None),
    # Нет букв -> detect вернет "", 0.0 -> дроп
    ("12345 !!!", "ru", "", 0.0, None),
], ids=["ru-ok", "wrong-lang", "low-conf", "no-letters"])
def test_language_id(monkeypatch, src, expected_lang, mock_detected, mock_conf, expected):
    # Мокаем функцию detect, чтобы не загружать fasttext
    monkeypatch.setattr(
        "dataset.cleaning.rules.detect",
        lambda text: (mock_detected, mock_conf)
    )
    result = language_id(src, expected_lang, min_confidence=0.8, sample_chars=2000)
    assert result == expected