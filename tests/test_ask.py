import pytest

from ragqa import ask
from ragqa.schemas import AnswerResult, Verification


def test_success_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["ragqa.ask", "質問"])
    monkeypatch.setattr(ask, "answer_question", lambda question: AnswerResult(
        question=question,
        answer="回答",
        verification=Verification(verdict="sufficient"),
        sources=[],
    ))
    assert ask.main() == 0
    output = capsys.readouterr()
    assert "回答" in output.out
    assert not output.err


@pytest.mark.parametrize("error", [
    FileNotFoundError("missing index"), RuntimeError("service failed"),
])
def test_runtime_failure_returns_one(monkeypatch, capsys, error):
    def fail(question):
        raise error

    monkeypatch.setattr("sys.argv", ["ragqa.ask", "質問"])
    monkeypatch.setattr(ask, "answer_question", fail)
    assert ask.main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert f"Error: {error}" in output.err


def test_missing_argument_returns_two_without_calling_service(monkeypatch, capsys):
    def unexpected_call(question):
        pytest.fail("Service must not run for a usage error")

    monkeypatch.setattr("sys.argv", ["ragqa.ask"])
    monkeypatch.setattr(ask, "answer_question", unexpected_call)
    assert ask.main() == 2
    assert 'Usage: python -m ragqa.ask "質問文"' in capsys.readouterr().err
