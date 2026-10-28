# Implementation log

## 2026-09-21 — Foundation

1. Created an installable Python project with FastAPI, LangGraph, Pydantic,
   SQLAlchemy, PostgreSQL driver, pytest, and Ruff.
2. Added environment-based database configuration and ignored local secrets,
   databases, caches, and virtual environments.
3. Added typed session, task, budget, source, section, document, claim, evidence,
   and timestamped event models. Claims default to insufficient evidence.
4. Added persistent JSON session snapshots using SQLAlchemy. SQLite supports
   local development; PostgreSQL is configurable through the same repository.
5. Added a real LangGraph workflow with a deterministic template planning node.
   It records six pending tasks and explicitly stops at `awaiting_providers`.
6. Added session creation/read/status, source/claim/evidence inspection, JSON
   export, deletion, and stored event replay. Unavailable report/continue
   operations return explicit conflicts instead of fabricated output.
7. Added an optional PostgreSQL/pgvector Compose service, bound to localhost.
8. Added API lifecycle, persistence, validation, isolation, and event replay tests.
9. Added local startup and PostgreSQL setup documentation.
10. Installed dependencies and generated `uv.lock` for reproducible environments.

### Validation

- Python 3.13.14: all 8 pytest tests passed.
- Ruff: all checks passed; source and tests formatted.
- Persistence was tested by reopening the app against the same SQLite database.
- A third-party Starlette/AnyIO deprecation warning remains; tests pass.
- PostgreSQL was not exercised: Docker is unavailable in this environment.
- Python 3.12 compatibility is declared but was not tested on a 3.12 interpreter.

### Scope and remaining work

This implements the foundation milestone, not the complete MVP. Budgets are
validated and stored; enforcement will be added with operations that consume them.
The graph currently runs synchronously and stores its final planning snapshot.
There is no background worker, crash-resume execution, LLM planning, authentication,
live streaming, relational evidence constraints, or database migration system yet.
PostgreSQL configuration is supplied but requires an actual database integration test.

Next implementation sequence:

- Tool registry, SearXNG search, bounded fetching, retries, search budget enforcement.
- arXiv and Semantic Scholar discovery and canonical source deduplication.
- Structured document reading, PDF parsing, section-aware chunks.
- Embeddings, pgvector retrieval, and ranking.
- Model provider, atomic claims, exact evidence links, verification and citation audit.
- Report synthesis, durable worker, and live updates in the evidence interface.
- Contradiction searches and an evaluation dataset before declaring the MVP complete.

## 2026-09-21 — Browser frontend

1. Added a responsive dashboard at `/`, served by the existing FastAPI process.
   Uses plain HTML/CSS/JavaScript rather than the proposed Streamlit UI; no Node
   runtime, frontend build, or additional server is required to run it.
2. Added a question form, suggested questions, validated budget inputs, and
   session creation connected to the real API.
3. Added `GET /research` with bounded pagination and newest-first ordering.
   The sidebar loads persisted session summaries, with refresh and load more.
4. Added session status, counts, plan, sources, claims/evidence, timeline, and
   report tabs. Evidence expands into source links, exact passages, section/page,
   confidence, evidence type, and extraction time when stored records exist.
5. Added session refresh, JSON download, confirmed deletion, selected-session
   URLs, request timeouts, error messages, and protection against stale selection
   responses. Content is rendered as text; source links allow HTTP(S) only.
6. Added mobile layouts, keyboard tab navigation, visible focus, form labels,
   and a live status region. Assets use system fonts and no external CDN.
7. Added API coverage for frontend assets and paginated, persistent history.

### Frontend validation

- All 9 pytest tests pass; Ruff and JavaScript syntax checks pass.
- Headless Chrome smoke check passes: creation, all tabs, reload, refresh,
   JSON download, deletion, literal HTML handling, and mobile overflow check.
- Browser checks use a temporary database and do not modify saved user sessions.
- Re-run with `uv run --with playwright python scripts/check_frontend.py`
   (requires installed Chrome). The script is optional development tooling.

### Next implementation steps

1. **Make discovery work:** tool registry, SearXNG plus arXiv/Semantic Scholar,
   deduplication, retries, and enforced search/source budgets.
2. **Read sources:** bounded URL fetching, PDF and HTML extraction, preserved
   sections/pages, document storage, and chunking.
3. **Connect a model:** configurable provider and structured planning/extraction,
   token accounting, plus retrieval to select relevant evidence.
4. **Verify findings:** atomic claims, exact evidence references, claim checking,
   contradiction searches, and citation auditing.
5. **Finish research execution:** durable background work, progress updates in
   this dashboard, continuation, and a cited report with explicit uncertainty.
6. **Validate the MVP:** PostgreSQL integration, schema migrations, a small
   human-checked research benchmark, then quality/cost/latency evaluation.

The frontend exposes current capability; it does not add search or generated
findings. Timeline refresh is manual, and reports display stored Markdown as
plain text. Authentication and multi-user access are still future work.

## 2026-09-21 — Source discovery

1. Added a capability-based registry and HTTP adapters for arXiv, Semantic Scholar,
   and optional SearXNG, with timeouts, response size limits, throttling, and retries.
2. Added abstracts/snippets, authors, dates, DOI/arXiv IDs, provider provenance,
   aliases, and deduplication by canonical IDs and URLs.
3. Persisted search queries, providers, task IDs, attempt status, results, errors,
   and timestamps. Every attempt consumes search budget; source limits cap records.
4. Added a single-process background worker. New sessions search automatically;
   busy sessions stay saved for manual continuation. Duplicate starts are rejected;
   deleting a running session does not recreate it.
5. Implemented continuation for pending, failed, and interrupted sessions, skipping
   successful queries. Restart recovery preserves charged attempts and saved results.
6. Added dashboard polling while searching, Continue discovery, search usage,
   real source metadata, and explicit distinction between metadata and evidence.
7. Added an optional SearXNG Compose service, JSON configuration, environment
   examples, setup documentation, and HTTPX as a runtime dependency.

### Discovery validation

- All 17 tests pass: parser fixtures, budgets, retries, partial failure, persistence,
   deduplication, restart recovery, busy admission, deletion, and continuation.
- JavaScript syntax, Ruff, and Chrome smoke checks pass. Browser checks disable
   network discovery; worker/API integration is tested with deterministic providers.
- Live API checks: arXiv's documented electron query returned 200; tested RAG
   queries returned 406. Anonymous Semantic Scholar returned 429. A successful live
   end-to-end discovery run was not verified; no sources or findings were fabricated.
- SearXNG parsing is fixture-tested; no live SearXNG or PostgreSQL service was available.

### Discovery limits and next milestone

Run one API process. This is not a distributed durable queue; graceful shutdown
waits for its run, and crash recovery requires continuation. Search metadata is
not evidence. Tasks remain pending for reading/analysis, and discovery completion
does not mean the question is answered. Earlier entries describe previous milestones.

Next: bounded HTML/PDF fetching and parsing with preserved sections/pages, then
model integration, evidence extraction, verification, and cited reports.

## 2026-09-21 — Provider troubleshooting

- Checked configuration without printing secrets: discovery enabled, Semantic
  Scholar key absent, SearXNG URL configured but not reachable in this environment.
- Simplified arXiv single-token syntax to the documented `all:term` form instead
  of individually quoting each token. The real `electron` adapter probe now
  returns one result; this does not establish that every topic query will succeed.
  The follow-up `retrieval augmented generation` probe still returned HTTP 406;
  the general arXiv access/query issue remains unresolved.
- Added `python -m researchos.doctor` for one-request-per-provider checks without
  creating sessions or printing credentials. Supports provider and query selection.
- Added actionable HTTP error messages. Rejected providers are skipped for the
  remainder of a run after their allowed retry rather than tried for every task.
- Handle numeric and HTTP-date Retry-After headers. Long cooldowns stop retries
  in the current run instead of retrying before the requested delay.
- Accept blank optional SearXNG/key environment values, update `.env.example`,
  and document key setup and explicit restart requirements. User `.env` unchanged.
- All 20 tests pass; Ruff passes. Anonymous Semantic Scholar remains HTTP 429;
  an authenticated check requires the user's own API key.

## 2026-10-08 — Document fetching and parsing

- Added `fetch.py` (bounded, SSRF-aware fetcher), `parse.py` (trafilatura and
  BeautifulSoup for HTML, PyMuPDF for PDF), and `reading.py` (single background
  reader, same pattern as discovery).
- New statuses `reading`, `reading_complete`, `reading_failed`; new
  `read_history` and `max_documents`; `Document` now holds metadata only and
  section text lives in a `document_sections` table. Existing sessions load
  unchanged; the new table is created on startup.
- Semantic Scholar searches now request `openAccessPdf` and store it as
  `pdf_url`. Sources discovered before this change have no `pdf_url`.
- Endpoints: `POST /research/{id}/read`, `GET /research/{id}/documents`,
  `GET /research/{id}/documents/{document_id}`.
- Found against live documents and fixed: PDF text lost word spaces when spaces
  were separate spans; a removed page that redirected to its site index was read
  as the source; the first HTML parser kept page chrome such as Wikipedia's
  language list, so trafilatura became the primary extractor.
- New dependencies: `beautifulsoup4`, `pymupdf`, `trafilatura`.
- 32 tests and Ruff pass. Live read of a saved 10-source session: 8 read, 2
  failed with reasons.

## 2026-10-08 — Model planning with Ollama (gpt-oss:120b)

- Added `llm.py` and `planning.py`; `Task.query`, `plan_source`; settings
  `OLLAMA_API_KEY`, `OLLAMA_BASE_URL`, `LLM_MODEL`; doctor `--provider llm`.
- Planning runs in the discovery worker before the first search, so creating a
  session stays fast. Template plan remains the fallback.
- Confirmed `gpt-oss:120b` is listed by `https://ollama.com/api/tags` and that
  `/api/chat` answers 401 without a key. No authenticated call has been made.
- 37 tests and Ruff pass (model behaviour simulated).
- Live check with a real key: doctor OK; plans valid on first attempt. Queries
  ran to 8-11 keywords, so they are now trimmed to six; rejecting them instead
  doubled token cost and once discarded a good plan. Full live session passed.
- Unattempted tasks now stay pending when a budget is exhausted.

## 2026-10-08 — Retrieval, claims, and evidence

- Added `embed.py`, `retrieval.py`, `analysis.py`; tables `document_chunks`;
  statuses `analyzing`, `analysis_complete`, `analysis_failed`; `Task.analysis`,
  `Claim.task_id`, `Evidence.chunk_id`/`heading`; `Evidence.confidence` is now
  optional. New dependency: `numpy`. `.env` gained `EMBEDDING_BASE_URL`.
- Live results and open issues are recorded in PROJECT_STATUS.md.
- Found live and fixed: PDF table fragments produced passages a few characters
  long; overlap could exceed the passage size limit.
- 46 tests and Ruff pass.
