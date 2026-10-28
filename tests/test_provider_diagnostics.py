from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from researchos.config import Settings
from researchos.discovery import Discovery
from researchos.models import ResearchRequest
from researchos.search import (
    SearchFailure,
    SemanticScholar,
    ToolRegistry,
    retry_delay,
)
from researchos.storage import Repository
from researchos.workflow import ResearchManager


def test_blank_optional_config(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "  ")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "  ")
    settings = Settings(_env_file=None)
    assert settings.searxng_url is None
    assert settings.semantic_scholar_api_key is None


def test_retry_after_is_not_shortened():
    assert retry_delay("120") == 120
    assert retry_delay("invalid") == 5
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=120))
    assert 118 < retry_delay(future) <= 120
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "120"})
        )
    ) as client:
        with pytest.raises(SearchFailure) as error:
            SemanticScholar(client, "private-test-key").search("electron", 1)
        assert not error.value.retryable
        assert error.value.suspend_provider
        assert "SEMANTIC_SCHOLAR_API_KEY" in str(error.value)
        assert "private-test-key" not in str(error.value)


def test_rejected_provider_is_not_repeated_for_every_task(tmp_path):
    class Rejected:
        name = "arxiv"
        source_type = "papers"
        calls = 0

        def search(self, query, limit):
            self.calls += 1
            raise SearchFailure("arxiv: HTTP 406", suspend_provider=True)

    repository = Repository(f"sqlite:///{tmp_path}/rejected.db")
    repository.initialize()
    provider = Rejected()
    discovery = Discovery(repository, ToolRegistry([provider]))
    state = ResearchManager(repository).create(ResearchRequest(question="RAG evaluation"))
    try:
        discovery.run(state)
        assert provider.calls == 1
        saved = repository.get(state.id)
        assert any(event.action == "provider_paused" for event in saved.events)
        assert len(saved.search_history) == 1
    finally:
        discovery.close()
        repository.engine.dispose()
