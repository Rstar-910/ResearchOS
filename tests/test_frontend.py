from fastapi.testclient import TestClient

from researchos.api import create_app
from researchos.config import Settings


def test_frontend_and_paginated_history(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ui.db'}", _env_file=None, discovery_enabled=False
    )
    with TestClient(create_app(settings)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "/static/app.js" in page.text
        for asset in ("app.js", "styles.css"):
            assert client.get(f"/static/{asset}").status_code == 200
        assert client.get("/research").json() == []
        first = client.post("/research", json={"question": "First research question"}).json()
        second = client.post("/research", json={"question": "Second research question"}).json()
        recent = client.get("/research?limit=1").json()
        assert recent == [{key: second[key] for key in ("id", "question", "status", "updated_at")}]
        assert client.get("/research?limit=1&offset=1").json()[0]["id"] == first["id"]
        assert client.get("/research?limit=0").status_code == 422
        assert client.get("/research?offset=-1").status_code == 422
    with TestClient(create_app(settings)) as client:
        assert len(client.get("/research").json()) == 2
        client.delete(f"/research/{second['id']}")
        assert [item["id"] for item in client.get("/research").json()] == [first["id"]]
