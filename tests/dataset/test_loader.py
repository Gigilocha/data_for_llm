import pytest
from dataset.loader import _make_id, load_documents


# Тест для функции _make_id
def test_make_id_stable():
    assert _make_id("ru_web", "42") == _make_id("ru_web", "42")
    assert _make_id("ru_web", "42") != _make_id("ru_web", "43")

def test_make_id_differs_by_source():
    assert _make_id("ru_web", "42") != _make_id("en_web", "42")

