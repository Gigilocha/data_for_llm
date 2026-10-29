import os
import pytest

MODEL_ENV = "FASTTEXT_LANG_MODEL"

@pytest.fixture(scope="session")
def lang_model_path() -> str:
    path = os.environ.get(MODEL_ENV)
    if not path or not os.path.isfile(path):
        pytest.skip(f"{MODEL_ENV} not set or file missing: {path!r}")
    os.environ[MODEL_ENV] = path
    return path