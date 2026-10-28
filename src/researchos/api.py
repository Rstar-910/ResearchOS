from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from researchos.analysis import Analyst
from researchos.config import Settings
from researchos.discovery import Discovery
from researchos.embed import Embedder, configured_embedder
from researchos.fetch import Fetcher
from researchos.llm import LLM, configured_llm
from researchos.models import ResearchRequest, ResearchState
from researchos.planning import Planner
from researchos.reading import Reader
from researchos.retrieval import search
from researchos.search import ToolRegistry, configured_registry
from researchos.storage import Repository
from researchos.workflow import ResearchManager


def create_app(
    settings: Settings | None = None,
    registry: ToolRegistry | None = None,
    fetcher: Fetcher | None = None,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    config = settings or Settings()
    repository = Repository(config.database_url)
    manager = ResearchManager(repository)
    client = httpx.Client(timeout=config.search_timeout, headers={"User-Agent": "ResearchOS/0.1"})
    llm = llm or configured_llm(config, client)
    discovery = Discovery(
        repository,
        registry or configured_registry(config, client),
        Planner(llm) if llm else None,
    )
    embedder = embedder or configured_embedder(config, client)
    reader = Reader(repository, fetcher or Fetcher(client), embedder)
    analyst = Analyst(repository, llm, embedder) if llm else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.initialize()
        repository.recover_interrupted()
        yield
        discovery.close()
        reader.close()
        if analyst:
            analyst.close()
        client.close()
        repository.engine.dispose()

    app = FastAPI(title="ResearchOS", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def frontend():
        return FileResponse(static_dir / "index.html")

    def get_session(research_id: UUID) -> ResearchState:
        session = repository.get(str(research_id))
        if session is None:
            raise HTTPException(404, "Research session not found")
        return session

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "stage": "analysis",
            "providers": [p.name for p in discovery.registry.providers],
            "model": llm.model if llm else None,
            "embedding_models": embedder.models if embedder else [],
        }

    @app.post("/research", response_model=ResearchState, status_code=201)
    def create_research(request: ResearchRequest):
        state = manager.create(request)
        if config.discovery_enabled:
            try:
                return discovery.start(state)
            except ValueError as error:
                state.status = "awaiting_providers"
                state.record("discovery_pending", str(error))
                repository.save(state)
        return state

    @app.get("/research")
    def list_research(
        limit: int = Query(default=30, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        return repository.list_sessions(limit, offset)

    @app.get("/research/{research_id}", response_model=ResearchState)
    def research(research_id: UUID):
        return get_session(research_id)

    @app.get("/research/{research_id}/status")
    def status(research_id: UUID):
        session = get_session(research_id)
        return {
            "id": session.id,
            "status": session.status,
            "current_subtask": None,
            "tasks": len(session.tasks),
            "sources": len(session.sources),
            "documents": len(session.documents),
            "claims": len(session.claims),
            "token_usage": session.token_usage,
            "searches_used": len(session.search_history),
            "errors": session.errors,
            "latest_event": session.events[-1] if session.events else None,
        }

    @app.get("/research/{research_id}/sources")
    def sources(research_id: UUID):
        return get_session(research_id).sources

    @app.get("/research/{research_id}/claims")
    def claims(research_id: UUID):
        return get_session(research_id).claims

    @app.get("/research/{research_id}/evidence")
    def evidence(research_id: UUID):
        return get_session(research_id).evidence

    @app.get("/research/{research_id}/report")
    def report(research_id: UUID):
        session = get_session(research_id)
        if session.report is None:
            raise HTTPException(409, "No report is available; research has not completed")
        return {"format": "markdown", "content": session.report}

    @app.post("/research/{research_id}/continue", status_code=202, response_model=ResearchState)
    def continue_research(research_id: UUID):
        try:
            return discovery.start(get_session(research_id))
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @app.post("/research/{research_id}/read", status_code=202, response_model=ResearchState)
    def read_sources(research_id: UUID):
        try:
            return reader.start(get_session(research_id))
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @app.post("/research/{research_id}/analyze", status_code=202, response_model=ResearchState)
    def analyze(research_id: UUID):
        if analyst is None:
            raise HTTPException(409, "No model is configured; set OLLAMA_API_KEY to analyze.")
        try:
            return analyst.start(get_session(research_id))
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/research/{research_id}/documents")
    def documents(research_id: UUID):
        return get_session(research_id).documents

    @app.get("/research/{research_id}/documents/{document_id}")
    def document(research_id: UUID, document_id: UUID):
        session = get_session(research_id)
        found = next((d for d in session.documents if d.id == str(document_id)), None)
        if found is None:
            raise HTTPException(404, "Document not found")
        return {"document": found, "sections": repository.get_sections(session.id, found.id)}

    @app.get("/research/{research_id}/search")
    def search_passages(
        research_id: UUID,
        q: str = Query(min_length=2, max_length=500),
        limit: int = Query(default=8, ge=1, le=30),
    ):
        return search(repository, embedder, get_session(research_id), q, limit)

    @app.post("/research/{research_id}/export")
    def export(research_id: UUID):
        session = get_session(research_id)
        return Response(
            session.model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="research-{session.id}.json"'},
        )

    @app.get("/research/{research_id}/stream")
    def stream(research_id: UUID, after: int = Query(default=0, ge=0)):
        session = get_session(research_id)

        def events():
            for event in session.events:
                if event.sequence > after:
                    yield (
                        f"id: {event.sequence}\nevent: progress\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )
            yield f'event: snapshot_end\ndata: {{"status":"{session.status}"}}\n\n'

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.delete("/research/{research_id}", status_code=204)
    def delete_research(research_id: UUID):
        if not repository.delete(str(research_id)):
            raise HTTPException(404, "Research session not found")
        return Response(status_code=204)

    return app
