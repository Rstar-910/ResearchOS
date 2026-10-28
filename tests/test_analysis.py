import json
import time

import httpx
from fastapi.testclient import TestClient

from researchos.analysis import quote_in_passage
from researchos.api import create_app
from researchos.config import Settings
from researchos.fetch import Fetcher
from researchos.llm import LLM
from researchos.models import ResearchState, Source, Task
from researchos.search import ToolRegistry
from researchos.storage import Repository

FINDING = "Small judges reached 0.61 agreement with human raters on the benchmark."
FILLER = "The study compared several evaluation models across many retrieval tasks. " * 4
PAGE = f"<html><body><article><h1>Results</h1><p>{FILLER}{FINDING}</p></article></body></html>"


class Provider:
    name = "fake"
    source_type = "papers"

    def search(self, query, limit):
        return [
            Source(
                title="Judge study",
                url="https://example.org/study",
                provider="fake",
                source_type="web",
            )
        ]


def make_app(tmp_path, name, llm_handler=None, budget_db=None):
    def pages(request):
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})

    settings = Settings(database_url=f"sqlite:///{tmp_path}/{name}.db", _env_file=None)
    fetcher = Fetcher(
        httpx.Client(transport=httpx.MockTransport(pages)), lambda h: ["93.184.216.34"], 0
    )
    llm = None
    if llm_handler:
        client = httpx.Client(transport=httpx.MockTransport(llm_handler))
        llm = LLM(client, "https://ollama.example", "gpt-oss:120b", "secret")
    return create_app(settings, ToolRegistry([Provider()]), fetcher, llm=llm)


def reply(payload):
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(payload)},
            "prompt_eval_count": 400,
            "eval_count": 100,
        },
    )


def model(request):
    body = json.loads(request.content)
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    if system.startswith("You plan"):
        return httpx.Response(500)
    if system.startswith("You extract"):
        assert "[1] Source: Judge study | Section: Results" in user and FINDING in user
        return reply(
            {
                "claims": [
                    {
                        "text": "Small judges reached 0.61 agreement with humans.",
                        "passage": 1,
                        "quote": FINDING,
                    },
                    {
                        "text": "Small judges beat GPT-4.",
                        "passage": 1,
                        "quote": "Small judges outperformed GPT-4 everywhere.",
                    },
                    {
                        "text": "Cites a passage that was never shown.",
                        "passage": 9,
                        "quote": FINDING,
                    },
                ]
            }
        )
    assert "Claim 1:" in user and "Claim 2:" not in user
    return reply({"verdicts": [{"claim": 1, "status": "SUPPORTED", "reason": "Stated directly."}]})


def wait_idle(client, research_id):
    for _ in range(400):
        state = client.get(f"/research/{research_id}").json()
        if state["status"] not in {"searching", "reading", "analyzing"}:
            return state
        time.sleep(0.01)
    raise AssertionError("Worker did not finish")


def read_session(client, budget=None):
    payload = {
        "question": "Are small judges reliable?",
        "budget": {"max_searches": 1, **(budget or {})},
    }
    created = client.post("/research", json=payload).json()
    wait_idle(client, created["id"])
    client.post(f"/research/{created['id']}/read")
    assert wait_idle(client, created["id"])["status"] == "reading_complete"
    return f"/research/{created['id']}", created["id"]


def test_quotes_must_appear_in_the_cited_passage():
    passage = (
        "Judges show “position bias” in pairwise  comparisons. It shrinks when order is swapped."
    )
    assert quote_in_passage('Judges show "position bias" in pairwise comparisons.', passage)
    assert quote_in_passage("Judges show ... It shrinks when order is swapped.", passage)
    assert not quote_in_passage("It shrinks when order is swapped. ... Judges show", passage)
    assert not quote_in_passage("Judges show no position bias in pairwise comparisons.", passage)
    assert not quote_in_passage("...", passage) and not quote_in_passage("Judges", passage)


def test_analysis_keeps_only_claims_whose_quotes_are_real(tmp_path):
    with TestClient(make_app(tmp_path, "analysis", model)) as client:
        base, research_id = read_session(client)
        assert client.post(f"{base}/analyze").status_code == 202
        state = wait_idle(client, research_id)
        assert state["status"] == "analysis_complete"
        # Six template tasks retrieve the same passage; each keeps one real claim of three.
        assert len(state["claims"]) == len(state["evidence"]) == 6
        claim, evidence = state["claims"][0], state["evidence"][0]
        assert claim["verification"] == "SUPPORTED" and claim["task_id"] == state["tasks"][0]["id"]
        assert evidence["text"] == FINDING and evidence["heading"] == "Results"
        assert evidence["claim_id"] == claim["id"] and evidence["confidence"] is None
        assert evidence["document_id"] == state["documents"][0]["id"]
        assert all(task["analysis"] == "completed" for task in state["tasks"])
        assert state["token_usage"] == 6 * 2 * 500
        assert any("2 dropped" in event["message"] for event in state["events"])
        assert client.post(f"{base}/analyze").status_code == 409
        assert client.post(f"{base}/read").status_code == 409


def test_analysis_needs_a_model_read_documents_and_budget(tmp_path):
    with TestClient(make_app(tmp_path, "nomodel")) as client:
        base, _ = read_session(client)
        assert client.post(f"{base}/analyze").status_code == 409
    with TestClient(make_app(tmp_path, "budget", model)) as client:
        fresh = client.post(
            "/research", json={"question": "Not read yet", "budget": {"max_searches": 1}}
        ).json()
        wait_idle(client, fresh["id"])
        assert client.post(f"/research/{fresh['id']}/analyze").status_code == 409
        base, research_id = read_session(client, {"max_tokens": 5000})
        client.post(f"{base}/analyze")
        state = wait_idle(client, research_id)
        # 1,000 tokens per task; analysis stops once fewer than 4,000 remain.
        assert [task["analysis"] for task in state["tasks"]].count("completed") == 2
        assert any(event["action"] == "budget_exhausted" for event in state["events"])


def test_restart_recovers_interrupted_analysis(tmp_path):
    repository = Repository(f"sqlite:///{tmp_path}/restart.db")
    repository.initialize()
    state = ResearchState(
        question="RAG", status="analyzing", tasks=[Task(question="q", source_types=["web"])]
    )
    repository.create(state)
    repository.recover_interrupted()
    assert repository.get(state.id).status == "reading_complete"
    repository.engine.dispose()
