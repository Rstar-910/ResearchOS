import json
import time

import httpx
from fastapi.testclient import TestClient

from researchos.api import create_app
from researchos.config import Settings
from researchos.llm import LLM, configured_llm
from researchos.models import Source
from researchos.planning import Plan
from researchos.search import ToolRegistry

PLAN = {
    "tasks": [
        {
            "question": f"Subquestion {n} about judges",
            "query": f"llm judge topic{n}",
            "source_types": ["papers"],
        }
        for n in range(5)
    ]
}


def reply(content, prompt=100, output=50):
    return httpx.Response(
        200,
        json={"message": {"content": content}, "prompt_eval_count": prompt, "eval_count": output},
    )


def model(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLM(client, "https://ollama.example", "gpt-oss:120b", "secret")


class Provider:
    name = "fake"
    source_type = "papers"

    def __init__(self):
        self.queries = []

    def search(self, query, limit):
        self.queries.append(query)
        return [
            Source(
                title=query,
                url=f"https://example.org/{len(self.queries)}",
                provider="fake",
                source_type="paper",
            )
        ]


def wait_idle(client, research_id):
    for _ in range(300):
        state = client.get(f"/research/{research_id}").json()
        if state["status"] not in {"searching", "reading"}:
            return state
        time.sleep(0.01)
    raise AssertionError("Worker did not finish")


def test_structured_output_sends_schema_and_repairs_once():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url == "https://ollama.example/api/chat"
        assert request.headers["authorization"] == "Bearer secret"
        if len(requests) == 1:
            return reply('Here is the plan: {"tasks": []}')
        return reply("```json\n" + json.dumps(PLAN) + "\n```")

    plan, used, error = model(handler).structured("system", "question", Plan, 10_000)
    assert len(plan.tasks) == 5 and used == 300 and error == ""
    assert requests[0]["format"]["properties"]["tasks"]["type"] == "array"
    assert requests[0]["stream"] is False and requests[0]["model"] == "gpt-oss:120b"
    assert "rejected" in requests[1]["messages"][-1]["content"]


def test_structured_output_reports_failures_and_spent_tokens():
    plan, used, error = model(lambda request: reply("not json")).structured("s", "q", Plan, 10_000)
    assert plan is None and used == 300 and "required format" in error
    plan, used, error = model(lambda request: httpx.Response(401)).structured(
        "s", "q", Plan, 10_000
    )
    assert plan is None and used == 0 and "OLLAMA_API_KEY" in error
    plan, used, error = model(lambda request: reply("not json")).structured("s", "q", Plan, 100)
    assert plan is None and used == 150 and error == "token budget exhausted"
    assert (
        model(lambda request: reply("{}")).structured("s", "q", Plan, 0)[2]
        == "token budget exhausted"
    )


def test_configuration_selects_cloud_local_or_none():
    client = httpx.Client()
    assert configured_llm(Settings(_env_file=None), client) is None
    cloud = configured_llm(Settings(_env_file=None, ollama_api_key="k"), client)
    assert cloud.url == "https://ollama.com/api/chat" and cloud.model == "gpt-oss:120b"
    local = configured_llm(
        Settings(_env_file=None, ollama_base_url="http://localhost:11434"), client
    )
    assert local.url == "http://localhost:11434/api/chat" and local.headers == {}
    client.close()


def test_model_plan_drives_discovery_queries(tmp_path):
    provider = Provider()
    settings = Settings(database_url=f"sqlite:///{tmp_path}/plan.db", _env_file=None)
    app = create_app(
        settings, ToolRegistry([provider]), llm=model(lambda r: reply(json.dumps(PLAN)))
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["model"] == "gpt-oss:120b"
        created = client.post(
            "/research", json={"question": "Are small LLM judges reliable?"}
        ).json()
        assert created["plan_source"] == "template"
        state = wait_idle(client, created["id"])
        assert state["plan_source"] == "model" and state["token_usage"] == 150
        assert [task["query"] for task in state["tasks"]] == provider.queries
        assert provider.queries[0] == "llm judge topic0"
        assert all(task["status"] == "completed" for task in state["tasks"])
        assert any(event["action"] == "plan_ready" for event in state["events"])


def test_failed_or_unaffordable_planning_keeps_template_plan(tmp_path):
    provider = Provider()
    settings = Settings(database_url=f"sqlite:///{tmp_path}/fallback.db", _env_file=None)
    app = create_app(settings, ToolRegistry([provider]), llm=model(lambda r: httpx.Response(500)))
    with TestClient(app) as client:
        created = client.post("/research", json={"question": "RAG evaluation"}).json()
        state = wait_idle(client, created["id"])
        assert state["plan_source"] == "template" and len(state["tasks"]) == 6
        assert state["status"] == "discovery_complete" and provider.queries[0] == "RAG evaluation"
        assert any(event["action"] == "planning_fallback" for event in state["events"])
