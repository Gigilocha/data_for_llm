import pytest
from dataset.cleaning.rules import (
    unicode_nfc,
    strip_control,
    fix_encoding,
    normalize_newlines,
    spacelike,
    spaces,
    space_nl,
    newlines,
)

pytestmark = pytest.mark.unit


# --- unicode_nfc ---
@pytest.mark.parametrize("src, expected", [
    ("e\u0301", "é"),
    ("и\u0306", "й"),
    ("Привет, мир!", "Привет, мир!"),
    ("cafe\u0301", "café"),
], ids=["e+acute", "i+breve", "already-nfc", "cafe+acute"])
def test_unicode_nfc(src, expected):
    assert unicode_nfc(src) == expected


# --- strip_control ---
@pytest.mark.parametrize("src, expected", [
    ("Привет\u200B мир\uFEFF!", "Привет мир!"),
    ("abc\u00ADdef", "abcdef"),
    ("a\u200Eb", "ab"),
    ("a\u202Eb", "ab"),
    ("обычный текст", "обычный текст"),
    ("", None),
    ("   \u200B  ", None),
], ids=["zwsp+bom", "soft-hyphen", "lrm", "rlo", "noop", "empty", "spaces"])
def test_strip_control(src, expected):
    assert strip_control(src) == expected


# --- fix_encoding ---
@pytest.mark.parametrize("src, expected", [
    ("ÐŸÑ€Ð¸Ð²ÐµÑ‚", "Привет"),
    ("cafÃ©", "café"),
    ("Привет", "Привет"),
    ("", None),
], ids=["ru-mojibake", "cafe-mojibake", "ru-ok", "empty"])
def test_fix_encoding(src, expected):
    assert fix_encoding(src) == expected


# --- normalize_newlines ---
@pytest.mark.parametrize("src, expected", [
    ("a\r\nb", "a\nb"),
    ("a\rb", "a\nb"),
    ("a\r\n\r\nb", "a\n\nb"),
    ("a\nb", "a\nb"),
    ("\r\n\r\n", None),
], ids=["crlf", "cr", "double-crlf", "noop", "only-newlines"])
def test_normalize_newlines(src, expected):
    assert normalize_newlines(src) == expected


# --- spacelike ---
@pytest.mark.parametrize("src, expected", [
    ("Цена\u00A0100\u2009рублей", "Цена 100 рублей"),
    ("a\u3000b", "a b"),
    ("a\u202Fb", "a b"),
    ("a\u2000\u2001\u2002b", "a   b"),
    ("обычный текст", "обычный текст"),
    ("\u00A0\u2009\u3000", None),
], ids=["nbsp+thin", "ideographic", "narrow-nbsp", "three-spaces", "noop", "only-spaces"])
def test_spacelike(src, expected):
    assert spacelike(src) == expected


# --- spaces ---
@pytest.mark.parametrize("src, expected", [
    ("a    b", "a b"),
    ("a\t\tb", "a b"),
    ("a  \t  b", "a b"),
    ("a b", "a b"),
    ("   ", None),
], ids=["many-spaces", "tabs", "mixed", "noop", "only-spaces"])
def test_spaces(src, expected):
    assert spaces(src) == expected


# --- space_nl ---
@pytest.mark.parametrize("src, expected", [
    ("a   \n   b", "a\nb"),
    ("a \n b", "a\nb"),
    ("a\nb", "a\nb"),
    ("a  \n\n  b", "a\n\nb"),
    ("   \n   ", None),
], ids=["spaces-around-nl", "single-space", "noop", "double-nl", "only-ws"])
def test_space_nl(src, expected):
    assert space_nl(src) == expected


# --- newlines ---
@pytest.mark.parametrize("src, expected", [
    ("a\n\n\nb", "a\n\nb"),
    ("a\n\n\n\n\nb", "a\n\nb"),
    ("a\n\nb", "a\n\nb"),
    ("a\nb", "a\nb"),
], ids=["3-to-2", "5-to-2", "2-noop", "1-noop"])
def test_newlines(src, expected):
    assert newlines(src) == expected