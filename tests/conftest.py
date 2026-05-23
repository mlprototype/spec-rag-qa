import pytest


@pytest.fixture(autouse=True)
def enforce_offline_mode(monkeypatch):
    """
    全テストでHuggingFaceへのネットワークアクセスを禁止する。
    """
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
