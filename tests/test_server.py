import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from ragqa import server
from ragqa.schemas import AnswerResult, Verification


def _answer(question):
    return AnswerResult(
        question=question,
        answer="回答",
        verification=Verification(verdict="sufficient", confidence=100),
        sources=[],
    )


def test_chat_endpoint_returns_answer(monkeypatch):
    monkeypatch.setattr(server, "answer_question", _answer)
    with TestClient(server.app) as client:
        response = client.post("/api/v1/chat", json={"query": "質問"})
    assert response.status_code == 200
    assert response.json() == _answer("質問").model_dump()


@pytest.mark.parametrize("error,status", [
    (FileNotFoundError("missing index"), 503),
    (RuntimeError("service failed"), 500),
])
def test_chat_endpoint_error_status(monkeypatch, error, status):
    def fail(question):
        raise error

    monkeypatch.setattr(server, "answer_question", fail)
    with TestClient(server.app) as client:
        response = client.post("/api/v1/chat", json={"query": "質問"})
    assert response.status_code == status
    assert response.json()["detail"] == (
        "Index not found. Please run ingest first." if status == 503 else str(error)
    )


def test_blocking_chat_requests_run_concurrently_off_event_loop(monkeypatch):
    barrier = threading.Barrier(2, timeout=3)
    event_loop_thread = threading.get_ident()
    worker_threads = []

    def blocking_answer(question):
        worker_threads.append(threading.get_ident())
        # Both requests must reach the service before either can finish.
        barrier.wait()
        return _answer(question)

    monkeypatch.setattr(server, "answer_question", blocking_answer)

    async def request_pair():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await asyncio.gather(
                client.post("/api/v1/chat", json={"query": "a"}),
                client.post("/api/v1/chat", json={"query": "b"}),
            )

    responses = asyncio.run(request_pair())
    assert [response.status_code for response in responses] == [200, 200]
    assert len(set(worker_threads)) == 2
    assert event_loop_thread not in worker_threads
