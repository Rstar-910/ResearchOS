import time

import httpx
import numpy as np
from fastapi.testclient import TestClient

from researchos.api import create_app
from researchos.config import Settings
from researchos.embed import Embedder, configured_embedder
from researchos.models import Document, Section, Source
from researchos.retrieval import TARGET_CHARS, chunk_sections, keyword_ranking
from researchos.search import ToolRegistry

TOPICS = {
    "judge": "LLM judges show position bias when grading answers. " * 6,
    "retrieval": "Dense retrieval recall is measured with relevance labels. " * 6,
}


def page(text):
    return f"<html><body><article><h1>Title</h1><p>{text}</p></article></body></html>"


def fake_ollama(down=None):
    """Embeds text on two axes: mentions of grading, and mentions of retrieval."""

    def handler(request):
        if request.url.host == "example.org":
            topic = request.url.path.strip("/")
            return httpx.Response(
                200, text=page(TOPICS[topic]), headers={"content-type": "text/html"}
            )
        import json

        body = json.loads(request.content)
        if down and body["model"] in down:
            return httpx.Response(404)
        vectors = [
            [
                1.0 + text.lower().count("bias") + text.lower().count("unfair"),
                1.0 + text.lower().count("retrieval"),
            ]
            for text in body["input"]
        ]
        return httpx.Response(200, json={"embeddings": vectors})

    return httpx.Client(transport=httpx.MockTransport(handler))


class Provider:
    name = "fake"
    source_type = "papers"

    def search(self, query, limit):
        return [
            Source(
                title=name, url=f"https://example.org/{name}", provider="fake", source_type="web"
            )
            for name in TOPICS
        ]


def wait_idle(client, research_id):
    for _ in range(300):
        state = client.get(f"/research/{research_id}").json()
        if state["status"] not in {"searching", "reading"}:
            return state
        time.sleep(0.01)
    raise AssertionError("Worker did not finish")


def test_chunks_stay_inside_sections_and_respect_size():
    document = Document(source_id="s", content_hash="h", title="Paper")
    long_paragraph = "A sentence about evaluation. " * 120
    sections = [
        Section(heading="Intro", text="A short opening paragraph.\n\nA second paragraph.", page=1),
        Section(heading="Table", text="0.85", page=1),
        Section(heading="Method", text=long_paragraph.strip(), page=2),
    ]
    chunks = chunk_sections(document, sections)
    assert chunks[0].text == "A short opening paragraph.\n\nA second paragraph."
    assert chunks[0].page == 1 and not any(chunk.heading == "Table" for chunk in chunks)
    method = [chunk for chunk in chunks if chunk.heading == "Method"]
    assert len(method) > 1 and all(chunk.page == 2 for chunk in method)
    assert all(len(chunk.text) <= TARGET_CHARS for chunk in chunks)
    assert all(chunk.section_id in {s.id for s in sections} for chunk in chunks)


def test_keyword_ranking_prefers_rare_matching_terms():
    texts = ["cats and dogs", "position bias in judges", "judges and dogs and cats and judges"]
    assert keyword_ranking("position bias", texts) == [1]
    assert keyword_ranking("judges", texts)[0] == 2
    assert keyword_ranking("unrelated", texts) == []


def test_embedder_normalizes_prefixes_and_falls_back():
    seen = []

    def handler(request):
        import json

        body = json.loads(request.content)
        seen.append(body)
        if body["model"] == "qwen3-embedding:0.6b":
            return httpx.Response(404)
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * len(body["input"])})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = Embedder(
        client, "http://localhost:11434", ["qwen3-embedding:0.6b", "nomic-embed-text"]
    )
    assert embedder.choose() == "nomic-embed-text"
    vectors = embedder.embed("nomic-embed-text", ["a"] * 20, query=True)
    assert vectors.shape == (20, 2) and np.allclose(vectors[0], [0.6, 0.8])
    assert seen[-1]["input"][0] == "search_query: a" and len(seen[-2]["input"]) == 16
    assert configured_embedder(Settings(_env_file=None), client) is None


def run_session(tmp_path, embedder, name):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/{name}.db", _env_file=None)
    client = fake_ollama()
    from researchos.fetch import Fetcher

    fetcher = Fetcher(client, lambda host: ["93.184.216.34"], 0)
    return create_app(settings, ToolRegistry([Provider()]), fetcher, embedder=embedder)


def test_reading_indexes_documents_and_hybrid_search_finds_meaning(tmp_path):
    embedder = Embedder(fake_ollama(), "http://localhost:11434", ["qwen3-embedding:0.6b"])
    with TestClient(run_session(tmp_path, embedder, "hybrid")) as client:
        created = client.post(
            "/research", json={"question": "LLM judges", "budget": {"max_searches": 1}}
        ).json()
        base = f"/research/{created['id']}"
        wait_idle(client, created["id"])
        client.post(f"{base}/read")
        state = wait_idle(client, created["id"])
        assert state["embedding_model"] == "qwen3-embedding:0.6b"
        assert all(d["chunk_count"] >= 1 and d["embedding_model"] for d in state["documents"])
        assert client.post(f"{base}/read").status_code == 409
        # "unfair" appears in no document; only the semantic side can match it.
        found = client.get(f"{base}/search", params={"q": "unfair unfair unfair"}).json()
        assert found["mode"] == "hybrid" and found["results"][0]["source_title"] == "judge"
        exact = client.get(f"{base}/search", params={"q": "relevance labels"}).json()
        assert exact["results"][0]["source_title"] == "retrieval"
        assert exact["results"][0]["chunk"]["heading"] == "Title"
        assert client.get(f"{base}/search", params={"q": "x"}).status_code == 422
        assert client.delete(base).status_code == 204


def test_without_embeddings_search_is_keyword_only_and_can_be_upgraded(tmp_path):
    down = Embedder(
        fake_ollama(down={"qwen3-embedding:0.6b"}),
        "http://localhost:11434",
        ["qwen3-embedding:0.6b"],
    )
    with TestClient(run_session(tmp_path, down, "keyword")) as client:
        created = client.post(
            "/research", json={"question": "LLM judges", "budget": {"max_searches": 1}}
        ).json()
        base = f"/research/{created['id']}"
        wait_idle(client, created["id"])
        client.post(f"{base}/read")
        state = wait_idle(client, created["id"])
        assert state["embedding_model"] is None
        assert all(
            d["chunk_count"] >= 1 and d["embedding_model"] is None for d in state["documents"]
        )
        found = client.get(f"{base}/search", params={"q": "position bias"}).json()
        assert found["mode"] == "keyword" and found["results"][0]["source_title"] == "judge"
        assert any(e["action"] == "index_error" for e in state["events"])
    working = Embedder(fake_ollama(), "http://localhost:11434", ["qwen3-embedding:0.6b"])
    with TestClient(run_session(tmp_path, working, "keyword")) as client:
        # Documents are already read; a second pass only adds the missing embeddings.
        assert client.post(f"{base}/read").status_code == 202
        state = wait_idle(client, created["id"])
        assert (
            len(state["read_history"]) == 2 and state["embedding_model"] == "qwen3-embedding:0.6b"
        )
        assert client.get(f"{base}/search", params={"q": "unfair"}).json()["mode"] == "hybrid"
