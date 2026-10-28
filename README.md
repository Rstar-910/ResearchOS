# ResearchOS

An evidence-first research system, being implemented from
`ResearchOS_Full_Structural_Plan.txt`.

## Current milestone: from question to quoted claims

ResearchOS creates a template plan, searches arXiv and Semantic Scholar, and saves
deduplicated source metadata. Configure SearXNG to include web search. The dashboard
refreshes automatically during discovery and displays provider errors and results.

After discovery, **Read sources** fetches each source and parses it into sections
(HTML by heading, PDF by heading and page). Section text is viewable per source.

With `OLLAMA_API_KEY` set, a model plans the research and **Analyze evidence**
extracts claims, each quoting the passage it cites. With `EMBEDDING_BASE_URL` set
to a local Ollama server, passages are searchable by meaning as well as keyword.
No report is written yet. Abstracts, snippets, and parsed text are untrusted input.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run uvicorn researchos.api:create_app --factory --reload --host 127.0.0.1
```

Open **http://127.0.0.1:8000** for the frontend, or http://127.0.0.1:8000/docs
for the interactive API. SQLite persists sessions
in `researchos.db` by default. No LLM key is needed. Academic APIs may restrict anonymous requests.

In the dashboard, enter a question (or select a suggested question), optionally
adjust the budget, and click **Start research**. Reopen saved sessions from
the sidebar. Tabs show the plan, sources, claims and evidence, saved timeline,
and report. **Refresh** loads the latest snapshot; **Export session** downloads
JSON. Deletion asks for confirmation. The URL preserves the selected session
when you reload. There is no separate frontend server or JavaScript build step.
The frontend uses HTML, CSS, and JavaScript served by FastAPI, instead of Streamlit,
to keep this initial dashboard in the same process and support a custom layout.

Create a session through the API:

```sh
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the main approaches to RAG evaluation?"}'
```

Use the returned `id` with `GET /research/{id}` and
`GET /research/{id}/status`. `GET /research/{id}/stream?after=0` replays stored
events and closes with `snapshot_end`; it is not yet a live worker stream.
`POST /research/{id}/export` downloads the full session as JSON.
`DELETE /research/{id}` permanently deletes that local session.
`POST /research/{id}/continue` starts eligible saved sessions (HTTP 202). Finished, busy,
or budget-exhausted sessions return HTTP 409. Reports are not generated yet.

## Search configuration

Copy `.env.example` to `.env` if you do not already have one:

- `DISCOVERY_ENABLED=true`: automatically search for new sessions.
- `SEMANTIC_SCHOLAR_API_KEY`: optional key for academic API access/rate limits.
- `SEARXNG_URL`: optional web search instance; unset means academic-only discovery.
- `SEARCH_TIMEOUT=15`: per-request timeout in seconds.

For local web search with Docker:

```sh
docker compose up -d searxng
```

Set `SEARXNG_URL=http://localhost:8080` in `.env` and restart the API. The supplied
`config/searxng.yml` enables JSON; its fixed secret and disabled limiter are for
localhost development only. Docker is not needed to use academic APIs or an
existing SearXNG server.

New sessions start discovery. Use **Continue discovery** for old pending,
interrupted, or failed sessions. Successful provider/query pairs are skipped on
continuation. Each request attempt, including retries, consumes the original
search budget. The source budget caps unique saved records. Each provider/query
gets at most two attempts per run, with throttling and bounded retry delays.

Run **one API process**, without multiple Uvicorn workers or replicas. One session
searches at a time; sessions submitted while busy remain saved for manual
continuation. Graceful shutdown waits for the run; crash recovery marks interrupted
sessions for continuation. This is not a distributed durable queue. The dashboard
polls every two seconds during discovery; the SSE endpoint still replays snapshots.

Queries are deterministic keyword queries sent to configured providers.
Domain-aware planning, source ranking, and model token accounting
are future work.

Reading fetches only public `http(s)` addresses on standard ports, re-checks every
redirect, and caps downloads at 20 MB and PDFs at 100 pages. Sources that cannot be
read (paywalls, bot challenges, JavaScript-only pages, scanned PDFs) are listed with
the reason and can be retried. PDF parsing uses PyMuPDF, which is AGPL-licensed.

Live checks on September 21, 2026: arXiv returned 406 for tested RAG queries (its
documented electron example returned 200); anonymous Semantic Scholar returned
429. Provider access is not guaranteed. Errors are recorded and other providers
continue. No live SearXNG instance was available for verification.

Provider references: [arXiv API manual](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md),
[Semantic Scholar API](https://api.semanticscholar.org/api-docs), and
[SearXNG search API](https://github.com/searxng/searxng/blob/master/docs/dev/search_api.rst).

## Search troubleshooting

For search troubleshooting, first run:

```sh
uv run python -m researchos.doctor --provider arxiv --query electron
uv run python -m researchos.doctor --provider semantic_scholar --query "RAG evaluation"
uv run python -m researchos.doctor --provider searxng --query "RAG evaluation"
```

These diagnostics print key presence, not key contents, and do not change sessions.
The all-provider default is `uv run python -m researchos.doctor`.
An absent Semantic Scholar key leaves requests subject to anonymous throttling;
request one at https://www.semanticscholar.org/product/api#api-key-form and set
`SEMANTIC_SCHOLAR_API_KEY` locally. No arXiv API key is consumed by this adapter.
Blank optional values are accepted. Restart the API after changing `.env`;
Uvicorn's Python reload does not reliably pick up `.env` changes. Create a fresh
session if the old session's search budget is exhausted.

SearXNG requires a reachable running server, not just a URL in `.env`. If using
the included Compose service, start it before setting `SEARXNG_URL`. Otherwise
leave `SEARXNG_URL=` empty to disable web search. Increasing `SEARCH_TIMEOUT`
can help slow responses but cannot fix HTTP 406 or 429.

## PostgreSQL setup

With Docker installed:

```sh
docker compose up -d postgres
cp .env.example .env
```

Set `DATABASE_URL` in `.env` to the PostgreSQL example, then start the API.
The Compose credentials are for local development. The database binds to
localhost. Changing database URLs does not migrate existing SQLite sessions.
The image includes pgvector, but embeddings and vector indexes are not implemented.
Tables are created on startup; schema migrations will be needed before evolving
an existing database schema. Session data is currently stored as JSON snapshots.

## Checks

```sh
uv run pytest
uv run ruff check .
```

Optional browser check with installed Google Chrome:

```sh
uv run --with playwright python scripts/check_frontend.py
```

This uses a temporary database to exercise creation, tabs, reload, refresh,
export, deletion, and mobile overflow without changing your saved research.

See [the implementation log](docs/IMPLEMENTATION_LOG.md) for completed work,
validation, limitations, and the next milestones. This API is a local development
system: authentication and a durable external worker are not implemented.

Implementation references: [LangGraph StateGraph](https://reference.langchain.com/python/langgraph/graph/state)
and [FastAPI lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/).
