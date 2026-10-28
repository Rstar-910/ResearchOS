from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from researchos.api import create_app
from researchos.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}", _env_file=None, discovery_enabled=False
    )


def test_session_lifecycle_and_restart(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/research", json={"question": "  How does   RAG work? "})
        assert response.status_code == 201
        session = response.json()
        research_id = session["id"]
        assert session["question"] == "How does RAG work?"
        assert session["status"] == "awaiting_providers"
        assert len(session["tasks"]) == 6
        assert all(task["status"] == "pending" for task in session["tasks"])
        assert session["token_usage"] == 0

    with TestClient(create_app(settings)) as client:
        base = f"/research/{research_id}"
        assert client.get(base).json() == session
        assert client.get(f"{base}/status").json()["status"] == "awaiting_providers"
        for resource in ("sources", "claims", "evidence"):
            assert client.get(f"{base}/{resource}").json() == []
        assert client.get(f"{base}/report").status_code == 409
        assert client.post(f"{base}/continue").status_code == 409
        assert client.get(base).json() == session
        assert client.post(f"{base}/export").json() == session
        stream = client.get(f"{base}/stream?after=1")
        assert "id: 1\n" not in stream.text
        assert "id: 2\n" in stream.text
        assert "event: snapshot_end" in stream.text
        assert client.delete(base).status_code == 204
        assert client.get(base).status_code == 404
        assert client.delete(base).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "hi"},
        {"question": "x" * 4001},
        {"question": "Valid question", "budget": {"max_searches": 0}},
        {"question": "Valid question", "unknown": True},
    ],
)
def test_invalid_requests(settings, payload):
    with TestClient(create_app(settings)) as client:
        assert client.post("/research", json=payload).status_code == 422


def test_missing_session_and_invalid_id(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/research/{uuid4()}").status_code == 404
        assert client.get("/research/not-a-uuid").status_code == 422
        assert client.get("/health").json()["status"] == "ok"


def test_sessions_are_isolated(settings):
    with TestClient(create_app(settings)) as client:
        first = client.post("/research", json={"question": "First question"}).json()
        second = client.post("/research", json={"question": "Second question"}).json()
        assert first["id"] != second["id"]
        assert {t["id"] for t in first["tasks"]}.isdisjoint(t["id"] for t in second["tasks"])
        client.delete(f"/research/{first['id']}")
        assert client.get(f"/research/{second['id']}").json() == second
