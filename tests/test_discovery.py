import time
from threading import Event

import httpx
from fastapi.testclient import TestClient

from researchos.api import create_app
from researchos.config import Settings
from researchos.models import Budget, ResearchState, SearchEvent, Source, Task
from researchos.search import (
    Arxiv,
    SearchFailure,
    Searxng,
    SemanticScholar,
    ToolRegistry,
    merge_sources,
)
from researchos.storage import Repository


class FakeProvider:
    name = "fake"
    source_type = "papers"

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def search(self, query, limit):
        self.calls += 1
        if self.fail:
            raise SearchFailure("fake: HTTP 429", retryable=True)
        return [
            Source(
                title="Test paper",
                url="https://example.org/paper",
                provider=self.name,
                source_type="paper",
                snippet="An abstract, not evidence.",
            )
        ]


def wait_finished(client, research_id):
    for _ in range(200):
        state = client.get(f"/research/{research_id}").json()
        if state["status"] != "searching":
            return state
        time.sleep(0.01)
    raise AssertionError("Worker did not finish")


def test_automatic_discovery_and_persistence(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/discovery.db", _env_file=None)
    provider = FakeProvider()
    with TestClient(create_app(settings, ToolRegistry([provider]))) as client:
        result = client.post(
            "/research",
            json={
                "question": "RAG evaluation",
                "budget": {
                    "max_searches": 3,
                    "max_sources": 2,
                },
            },
        )
        assert result.status_code == 201
        state = wait_finished(client, result.json()["id"])
        assert state["status"] == "discovery_complete"
        assert len(state["sources"]) == 1
        assert len(state["search_history"]) == provider.calls == 3
        assert state["report"] is None and state["claims"] == []
        assert client.post(f"/research/{state['id']}/continue").status_code == 409
    with TestClient(create_app(settings, ToolRegistry([provider]))) as client:
        assert client.get(f"/research/{state['id']}").json() == state


def test_retries_count_toward_budget_and_stop(tmp_path):
    provider = FakeProvider(fail=True)
    settings = Settings(database_url=f"sqlite:///{tmp_path}/failure.db", _env_file=None)
    with TestClient(create_app(settings, ToolRegistry([provider]))) as client:
        result = client.post(
            "/research",
            json={
                "question": "RAG evaluation",
                "budget": {
                    "max_searches": 1,
                },
            },
        )
        state = wait_finished(client, result.json()["id"])
        assert provider.calls == 1
        assert state["status"] == "discovery_failed"
        assert state["search_history"][0]["error"] == "fake: HTTP 429"
        assert client.post(f"/research/{state['id']}/continue").status_code == 409


def test_source_limit(tmp_path):
    provider = FakeProvider()
    settings = Settings(database_url=f"sqlite:///{tmp_path}/limit.db", _env_file=None)
    with TestClient(create_app(settings, ToolRegistry([provider]))) as client:
        result = client.post(
            "/research",
            json={
                "question": "RAG evaluation",
                "budget": {
                    "max_sources": 1,
                },
            },
        )
        state = wait_finished(client, result.json()["id"])
        assert len(state["sources"]) == provider.calls == 1


def test_deduplication_merges_provenance_and_preserves_identity():
    first = Source(
        title="Paper",
        url="https://arxiv.org/abs/2301.12345v1",
        provider="arxiv",
        source_type="paper",
    )
    second = Source(
        title="Paper",
        url="https://example.org/paper",
        provider="semantic_scholar",
        source_type="paper",
        arxiv_id="2301.12345",
        doi="10.1000/example",
    )
    third = Source(
        title="Paper", url="https://doi.org/10.1000/example", provider="searxng", source_type="web"
    )
    sources = [first]
    merge_sources(sources, [second, third], 1)
    assert len(sources) == 1 and sources[0].id == first.id
    assert set(sources[0].providers) == {"arxiv", "semantic_scholar", "searxng"}


def test_provider_parsers_and_request_configuration():
    def handler(request):
        if request.url.host == "export.arxiv.org":
            assert request.url.params["max_results"] == "2"
            assert request.url.params["search_query"] == "all:RAG AND all:evaluation"
            return httpx.Response(
                200,
                text="""<feed xmlns="http://www.w3.org/2005/Atom">
                <entry><id>https://arxiv.org/abs/2301.12345v1</id><title>Test paper</title>
                <summary>Abstract</summary><author><name>Ada</name></author></entry></feed>""",
            )
        if request.url.host == "api.semanticscholar.org":
            assert request.headers["x-api-key"] == "test-secret"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "title": "Paper",
                            "url": "https://example.org",
                            "externalIds": {"DOI": "10.1000/example"},
                            "authors": [{"name": "Ada"}],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Invalid", "url": "javascript:alert(1)"},
                    {"title": "Valid", "url": "https://example.org", "content": "Snippet"},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert Arxiv(client).search("RAG evaluation", 2)[0].authors == ["Ada"]
        assert SemanticScholar(client, "test-secret").search("rag", 2)[0].doi == "10.1000/example"
        results = Searxng(client, "https://search.example.org").search("rag", 2)
        assert len(results) == 1 and results[0].snippet == "Snippet"


def test_restart_recovers_inflight_attempt_without_resetting_budget(tmp_path):
    repository = Repository(f"sqlite:///{tmp_path}/restart.db")
    repository.initialize()
    task = Task(question="RAG", source_types=["papers"])
    state = ResearchState(
        question="RAG",
        status="searching",
        tasks=[task],
        budget=Budget(max_searches=2),
        search_history=[SearchEvent(provider="arxiv", query="RAG", task_id=task.id)],
    )
    repository.create(state)
    repository.recover_interrupted()
    loaded = repository.get(state.id)
    assert loaded.status == "interrupted"
    assert len(loaded.search_history) == 1
    assert loaded.search_history[0].status == "failed"
    repository.engine.dispose()


def test_retry_and_other_provider_success(tmp_path):
    failed = FakeProvider(fail=True)
    failed.name = "unavailable"
    working = FakeProvider()
    settings = Settings(database_url=f"sqlite:///{tmp_path}/partial.db", _env_file=None)
    with TestClient(create_app(settings, ToolRegistry([failed, working]))) as client:
        result = client.post("/research", json={"question": "RAG", "budget": {"max_searches": 3}})
        state = wait_finished(client, result.json()["id"])
        assert failed.calls == 2 and working.calls == 1
        assert state["status"] == "discovery_complete"
        assert len(state["errors"]) == 2 and len(state["sources"]) == 1


def test_busy_admission_and_delete_during_search(tmp_path):
    entered, release = Event(), Event()

    class SlowProvider(FakeProvider):
        def search(self, query, limit):
            entered.set()
            assert release.wait(5)
            return super().search(query, limit)

    settings = Settings(database_url=f"sqlite:///{tmp_path}/busy.db", _env_file=None)
    with TestClient(create_app(settings, ToolRegistry([SlowProvider()]))) as client:
        first = client.post("/research", json={"question": "RAG"}).json()
        try:
            assert entered.wait(2)
            assert client.post(f"/research/{first['id']}/continue").status_code == 409
            second = client.post("/research", json={"question": "Another question"}).json()
            assert second["status"] == "awaiting_providers"
            assert client.delete(f"/research/{first['id']}").status_code == 204
        finally:
            release.set()
    with TestClient(create_app(settings, ToolRegistry([FakeProvider()]))) as client:
        assert client.get(f"/research/{first['id']}").status_code == 404
        response = client.post(f"/research/{second['id']}/continue")
        assert response.status_code == 202
        assert wait_finished(client, second["id"])["sources"]
