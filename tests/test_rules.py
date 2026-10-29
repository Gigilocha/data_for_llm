import pytest
from dataset.rules import (
    RE_CONTROL, RE_SPACELIKE, 
    RE_SPACES, RE_SPACE_NL, RE_NEWLINES, 
    RE_EMPTY_BRACKETS, RE_EMPTY_QUOTES, 
    RE_HTML_BLOCKS, RE_HTML_COMMENT, RE_HTML_TAG)
from dataset.rules import (
    unicode_nfc, strip_control, 
    fix_encoding, strip_html, 
    spacelike, spaces, space_nl, normalize_newlines,
    drop_empty_brackets, drop_empty_quotes, collapse_repeats,
    language_id, quality, length_gate, split_long)


# Тест unicode_nfc
@pytest.mark.parametrize("src, expected", [
    ("e\u0301", "é"),                     # NFC: "e" + combining acute → единый "é"
    ("и\u0306", "й"),                     # NFC: "и" + combining breve → "й"
    ("Привет, мир!", "Привет, мир!"),     # Уже в NFC — no-op
    ("cafe\u0301", "café"),               # Комбинированный диакритик в середине
], ids=["e+acute", "i+breve", "already-nfc", "cafe+acute"])
def test_unicode_nfc(src, expected):
    got = unicode_nfc(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест strip_control
@pytest.mark.parametrize("src, expected", [
    ("Привет\u200B мир\uFEFF!", "Привет мир!"),  # ZWSP + BOM удалены
    ("abc\u00ADdef", "abcdef"),                  # soft hyphen удалён
    ("a\u200Eb", "ab"),                          # LRM удалён
    ("a\u202Eb", "ab"),                          # RLO удалён
    ("обычный текст", "обычный текст"),          # no-op
    ("", None),                                  # пусто → None
    ("   \u200B  ", None),                       # пробельная пустота → None
], ids=["zwsp+bom", "soft-hyphen", "lrm", "rlo", "noop", "empty", "spaces"])
def test_strip_control(src, expected):
    got = strip_control(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест fix_encoding
@pytest.mark.parametrize("src, expected", [
    ("ÐŸÑ€Ð¸Ð²ÐµÑ‚", "Привет"), # классический utf-8 → latin-1
    ("cafÃ©", "café"),
    ("naÃ¯ve", "naïve"),
    ("Привет", "Привет"), # уже нормально — no-op
    ("café", "café"),
    ("hello world", "hello world"),
    ("", None), # пусто
    ("   ", None),
], ids=[
    "ru-mojibake", "cafe-mojibake", "naive-mojibake",
    "ru-ok", "cafe-ok", "ascii-ok",
    "empty", "spaces",
])
def test_fix_encoding(src, expected):
    got = fix_encoding(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест strip_html
@pytest.mark.parametrize("src, expected", [
    ("<p>Привет <b>мир</b></p>", "Привет мир"), # теги, содержимое остаётся
    ("<div><span>текст</span></div>", "текст"),
    ('<a href="https://x.y">ссылка</a>', "ссылка"),
    ("<br>", None),
    ("строка<br>перенос", "строкаперенос"),
    ('До<script>alert("x")</script>после', "Допосле"), # script/style — целиком
    ("<style>body{color:red}</style>текст", "текст"),
    ("a<!-- c -->b", "ab"), # комментарии
    ("До<!--\nмногострочный\n-->после", "Допосле"),
    ("<div><p><b>x</b></p></div>", "x"), # вложенность
    ("без тегов", "без тегов"), # no-op
    ("<p></p>", None), # пусто
    ("<!-- only -->", None),
], ids=[
    "p+b", "div+span", "a+href", "br-only", "text+br",
    "script", "style", "comment", "multiline-comment",
    "nested", "noop", "empty-p", "only-comment",
])
def test_strip_html(src, expected):
    got = strip_html(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест spacelike
@pytest.mark.parametrize("src, expected", [
    ("Цена\u00A0100\u2009рублей", "Цена 100 рублей"),      # NBSP + thin
    ("a\u3000b", "a b"),                                    # ideographic
    ("a\u202Fb", "a b"),                                    # narrow NBSP
    ("a\u2000\u2001\u2002b", "a   b"),                      # три разных → три пробела
    ("обычный текст", "обычный текст"),                     # no-op
    ("", None),                                              # пусто
    ("\u00A0\u2009\u3000", None),                           # только пробелы → None
], ids=[
    "nbsp+thin", "ideographic", "narrow-nbsp",
    "three-spaces", "noop", "empty", "only-spaces",
])
def test_spacelike(src, expected):
    got = spacelike(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест space_nl
@pytest.mark.parametrize("src, expected", [
    ("a   \n   b", "a\nb"),
    ("a \n b", "a\nb"),
    ("a\nb", "a\nb"),                     # no-op
    ("a  \n\n  b", "a\n\nb"),             # два перевода
    ("a\n\n\nb", "a\n\n\nb"),             # без пробелов — не трогается
    ("a\t\n\tb", "a\t\n\tb"),             # табы не трогаются
    ("a b", "a b"),                       # нет \n — no-op
    ("   \n   ", None),                   # только \n и пробелы → None
    ("", None),
], ids=[
    "spaces-around-nl", "single-space", "noop",
    "double-nl", "no-spaces", "tabs-kept",
    "no-newline", "only-ws", "empty",
])
def test_space_nl(src, expected):
    got = space_nl(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест normalize_newlines
@pytest.mark.parametrize("src, expected", [
    ("a\r\nb", "a\nb"),
    ("a\rb", "a\nb"),
    ("a\r\n\r\nb", "a\n\nb"),
    ("a\r\rb", "a\n\nb"),
    ("a\r\nb\rc\nd", "a\nb\nc\nd"),        # смесь
    ("a\nb", "a\nb"),                       # no-op
    ("a b", "a b"),                         # нет переводов
    ("\r\n\r\n", None),                     # только переводы → None
    ("", None),
], ids=[
    "crlf", "cr", "double-crlf", "double-cr",
    "mixed", "noop", "no-newlines",
    "only-newlines", "empty",
])
def test_normalize_newlines(src, expected):
    got = normalize_newlines(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест drop_empty_brackets
@pytest.mark.parametrize("src, expected", [
    # удаляются (перед скобкой не \w)
    ("текст () ещё", "текст  ещё"),
    ("текст [ ] ещё", "текст  ещё"),
    ("текст {  } ещё", "текст  ещё"),
    ("()", None),                          # только пустые скобки → None
    ("( ) [ ] { }", None),
    # НЕ удаляются (прилипли к слову)
    ("print()", "print()"),
    ("x = f()", "x = f()"),
    ("a[ ]", "a[ ]"),
    ("obj{}", "obj{}"),
    # НЕ пустые
    ("(a)", "(a)"),
    ("[x]", "[x]"),
    ("{y}", "{y}"),
    ("( a )", "( a )"),                    # внутри не только пробелы
    # no-op
    ("обычный текст", "обычный текст"),
    ("", None),
], ids=[
    "round", "square", "curly", "only-round", "only-all",
    "print", "f-call", "a-bracket", "obj-curly",
    "round-nonempty", "square-nonempty", "curly-nonempty", "spaced-nonempty",
    "noop", "empty",
])
def test_drop_empty_brackets(src, expected):
    got = drop_empty_brackets(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест drop_emtpy_quotes
@pytest.mark.parametrize("src, expected", [
    # удаляются
    ("Он сказал «» и всё", "Он сказал  и всё"),
    ("«»", None),
    ("«  »", None),
    ("„ “", None),
    ("“ ”", None),
    ("‘ ’", None),
    ("‹ ›", None),
    ("«» «» «»", None),                    # все пустые
    # прилипли к слову — тоже удаляются (нет (?<!\w))
    ("a«»b", "ab"),
    ("текст«»ещё", "текстещё"),
    # не пустые — остаются
    ("«ok»", "«ok»"),
    ("„слово“", "„слово“"),
    ("“цитата”", "“цитата”"),
    ("‘буква’", "‘буква’"),
    ("‹угол›", "‹угол›"),
    ("«  x  »", "«  x  »"),
    # no-op
    ("обычный текст", "обычный текст"),
    ('"ascii"', '"ascii"'),                # ASCII-кавычки не в паттерне
    ("", None),
], ids=[
    "ru-empty", "only-ru", "ru-spaces", "de-empty", "en-empty",
    "single-empty", "angle-empty", "many-empty",
    "glued", "glued-ru",
    "ru-nonempty", "de-nonempty", "en-nonempty", "single-nonempty", "angle-nonempty",
    "spaced-nonempty", "noop", "ascii-kept", "empty",
])
def test_drop_empty_quotes(src, expected):
    got = drop_empty_quotes(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест test_collapse_repeats
@pytest.mark.parametrize("src, expected", [
    ("Ну!!!!", "Ну!!!"), # сжимаются (max_run=3)
    ("Ну!!!!!", "Ну!!!"),
    ("Да--------", "Да---"),
    ("x====y", "x===y"),
    ("a....b", "a...b"),
    ("Что???", "Что???"), # 3 — не трогается
    ("a!!!!!b?????c", "a!!!b???c"),
    # не трогаются
    ("Ну!!!", "Ну!!!"),                # ровно 3
    ("a..b", "a..b"),                  # 2
    ("a.b", "a.b"),                    # 1
    ("a!?!", "a!?!"),                  # разные знаки
    ("(((", "((("),                     # не в классе
    ("###", "###"),
    ("обычный текст", "обычный текст"),
    ("", None),
], ids=[
    "4-to-3", "5-to-3", "dashes", "eq-4-to-3", "dots-4-to-3",
    "q-3-noop", "mixed-runs",
    "3-noop", "2-noop", "1-noop", "different-signs", "parens", "hash",
    "noop", "empty",
])
def test_collapse_repeats(src, expected):
    got = collapse_repeats(src, max_run=3)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест определения языка
RU = (
    "Москва — столица России, крупнейший по численности населения город "
    "страны и её субъект, самый населённый из городов, полностью "
    "расположенных в Европе."
)
EN = (
    "Moscow is the capital and largest city of Russia. The city stands on "
    "the Moskva River, and it is the political, economic, and cultural "
    "center of the country."
)

@pytest.mark.parametrize("src, expected_lang, expected", [
    (RU, "ru", RU),
    (RU, "en", None),
    (EN, "ru", None),
    ("12345 !!! 67890 ??? ---", "ru", None),
    ("", "ru", None),
], ids=["ru-as-ru", "ru-as-en", "en-as-ru", "digits", "empty"])
def test_language_id(lang_model_path, src, expected_lang, expected):
    got = language_id(src, expected_lang, min_confidence=0.5, sample_chars=2000)
    assert got == expected, f"\nsrc={src!r}\nexpected_lang={expected_lang!r}\ngot={got!r}"


# Тест качества текста
@pytest.mark.parametrize("src, expected", [
    # проходят (букв ≥ 60% видимых)
    ("Привет мир", "Привет мир"),
    ("Hello, world!", "Hello, world!"),
    ("abc", "abc"),
    ("a b c", "a b c"),                # пробелы не в visible
    # дропаются (букв < 60%)
    ("abc 123", None),                 # 3/6 = 0.5
    ("a1b2c3", None),                  # 3/6 = 0.5
    ("12345 !!!", None),               # 0/8 = 0.0
    ("!!!???", None),
    ("12345", None),
    ("", None),
    ("   ", None),                     # visible=1, 0/1=0
], ids=[
    "ru", "en", "letters-only", "spaced-letters",
    "half-letters", "alternating", "digits-punct", "punct-only",
    "digits-only", "empty", "spaces",
])
def test_quality(src, expected):
    got = quality(src)
    assert got == expected, f"\nsrc={src!r}\ngot={got!r}\nexp={expected!r}"


# Тест test_length_gate
@pytest.mark.parametrize("src, min_length, expected", [
    ("короткий текст", 256, None),           # 15 < 256
    ("a" * 255, 256, None),                   # граница: 255 < 256
    ("a" * 256, 256, "a" * 256),              # граница: 256 >= 256
    ("a" * 300, 256, "a" * 300),              # 300 > 256
    ("a" * 1000, 256, "a" * 1000),
    ("", 256, None),
    ("   \n   ", 256, None),
    # свой порог
    ("abc", 2, "abc"),                        # 3 >= 2
    ("abc", 4, None),                         # 3 < 4
], ids=[
    "short", "boundary-255", "boundary-256", "long", "very-long",
    "empty", "only-ws",
    "custom-pass", "custom-fail",
])
def test_length_gate(src, min_length, expected):
    got = length_gate(src, min_length=min_length)
    assert got == expected, f"\nlen(src)={len(src)}\nmin_length={min_length}\ngot={got!r}"


# Тест split_long
@pytest.mark.parametrize("src, max_length, on_too_long, expected", [
    # короткие — один чанк
    ("abcdef", 8, "split", ["abcdef"]),
    ("abcdefgh", 8, "split", ["abcdefgh"]),       # ровно max_length
    ("a" * 255, 256, "split", ["a" * 255]),
    # длинные — split
    ("abcdefghi", 8, "split", ["abcdefgh", "i"]),
    ("abcdefghij", 8, "split", ["abcdefgh", "ij"]),
    ("a" * 20, 8, "split", ["a" * 8, "a" * 8, "a" * 4]),
    ("a" * 17, 8, "split", ["a" * 8, "a" * 8, "a"]),
    # длинные — drop
    ("abcdefghi", 8, "drop", []),
    ("a" * 20, 8, "drop", []),
    # граница
    ("a" * 256, 256, "split", ["a" * 256]),
    ("a" * 257, 256, "split", ["a" * 256, "a"]),
], ids=[
    "short", "exact-max", "long-short",
    "just-over", "over-2", "many-chunks", "remainder",
    "drop-just-over", "drop-many",
    "boundary-256", "boundary-257",
])
def test_split_long(src, max_length, on_too_long, expected):
    got = split_long(src, max_length=max_length, on_too_long=on_too_long)
    assert got == expected, f"\nsrc={src!r}\nmax_length={max_length}\ngot={got!r}\nexp={expected!r}"