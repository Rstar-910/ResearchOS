# ResearchOS project status

Last updated: 2026-10-08 (retrieval and claim extraction verified live)

ResearchOS is currently a local, single-user prototype that runs from a question
to quoted, checked claims. It plans with a model, discovers sources, reads them
into sections with headings and page numbers, indexes them as searchable
passages, and extracts claims that each quote the passage they cite. It cannot
yet search for contradicting evidence or write a research report.

## Completed work

### Project foundation

- Python project setup with FastAPI, LangGraph, Pydantic, SQLAlchemy, HTTPX,
  pytest, and Ruff.
- Environment-based configuration with a committed `.env.example`; local `.env`
  files, databases, and virtual environments are ignored by Git.
- SQLite persistence for local development and configurable PostgreSQL support.
- Typed models for research sessions, tasks, budgets, sources, documents,
  sections, claims, evidence, events, and search attempts.
- Session creation, lookup, pagination, export as JSON, and deletion endpoints.
- Research event timeline and stored-server-sent-event replay endpoint.

### Planning and source discovery

- Deterministic six-part research plan based on the submitted question. It is a
  template plan; no LLM is used for planning.
- Background discovery worker for one session at a time in one API process.
- arXiv and Semantic Scholar adapters, plus optional SearXNG web discovery.
- Search request timeouts, provider throttling, 5 MB response limit, controlled
  retries, and persisted provider error messages.
- Search and source budgets: every request attempt consumes a search unit;
  only unique source records count towards the source limit.
- Deduplication using DOI, arXiv ID, and normalized URLs, retaining provider
  provenance and alternate links.
- Continuation of eligible saved sessions; successful provider/query pairs are
  skipped on continuation.
- Restart recovery marks a running session as interrupted so it can be continued
  without silently losing its recorded attempts.
- Provider diagnostic command: `uv run python -m researchos.doctor`.
- Task status now reflects real progress: each task moves `pending` ->
  `running` -> `completed`/`failed` as its providers are searched, instead of
  staying `pending` for the whole session (fixed in `discovery.py`). A task
  shows `failed` when every provider it depends on returned no results (for
  example, a papers-only task after both arXiv and Semantic Scholar are
  paused for the run) — this is an accurate status, not a bug.

### Document fetching and parsing

- Explicit reading stage after discovery: `POST /research/{id}/read`, or the
  **Read sources** button. It is not started automatically.
- Safe, bounded fetcher (`fetch.py`): public `http(s)` addresses on ports 80/443
  only, no credentials in URLs, every redirect hop re-validated, 20 MB cap,
  90-second deadline, content-type checks, and per-host pacing.
- Redirects to a parent/index page of the same site are treated as "page gone"
  rather than read as the source.
- HTML parsing (`parse.py`): trafilatura main-content extraction with a
  BeautifulSoup block parser as fallback; sections follow headings, and table
  rows are kept as ` | `-separated lines.
- PDF parsing: PyMuPDF; headings detected by font size; a section never spans a
  page, so every passage keeps an exact page number. Capped at 100 pages and
  500,000 characters, with a `truncated` flag.
- Full-text location per source: open-access PDF link from Semantic Scholar,
  then the arXiv PDF, then the source page, then the DOI. A landing page that
  advertises a `citation_pdf_url` is replaced by that PDF. Semantic Scholar's
  own paper pages are never read as documents.
- Document metadata (final URL, type, SHA-256 hash, pages, section and character
  counts) is stored in the session; section text is stored in a separate
  `document_sections` table and served by `GET /research/{id}/documents/{doc_id}`.
- Per-source read history: completed, duplicate (same content hash as a document
  already read), or failed with the reason. A second read retries only unread
  sources. `max_documents` budget (default 20).
- Restart during reading returns the session to a readable state; deleting a
  session removes its section text.

### Model integration (planning only)

- Ollama chat client (`llm.py`) for Ollama Cloud (`OLLAMA_API_KEY`) or a local
  Ollama server (`OLLAMA_BASE_URL`); default model `gpt-oss:120b` (`LLM_MODEL`).
- Structured output: the JSON schema is sent with the request, the reply is
  validated with Pydantic, and one repair attempt is made on invalid output.
- Question-specific planning (`planning.py`): when a model is configured, the
  discovery worker asks it for 5 to 8 subquestions, each with its own search
  query and source types, before the first search. On any failure the template
  plan is kept and the reason is recorded in the timeline.
- Token accounting: prompt and output tokens are added to `token_usage`, failed
  calls included, and no call is made once `max_tokens` is spent.
- `uv run python -m researchos.doctor --provider llm` checks model access.
- Search queries are trimmed to six keywords, because arXiv requires every
  keyword to match. The model is told today's date.
- Verified live on 2026-10-08 with `gpt-oss:120b` on Ollama Cloud: valid JSON on
  the first attempt, about 1,300 to 1,700 tokens and 3 to 5 seconds per plan.
  One full session: model plan, 20 sources from all three providers, 12 of 13
  attempted documents read (the document budget was 12).
- Tasks the search or source budget never reached now stay `pending` instead of
  being marked `failed`.

### Retrieval

- Section-aware chunking (`retrieval.py`): passages of up to 1,500 characters
  that never cross a section, so each keeps one heading and one page. Fragments
  under 40 characters are dropped.
- Embeddings from a local Ollama server (`embed.py`, `EMBEDDING_BASE_URL`):
  `qwen3-embedding:0.6b`, falling back to `nomic-embed-text` when a session's
  index is first built. A session uses one model throughout.
- Indexing runs at the end of the reading stage. Passages and vectors are stored
  in a `document_chunks` table. If embedding is unavailable, passages are still
  stored and a later run adds the vectors.
- Hybrid search (`GET /research/{id}/search`): cosine similarity and BM25 fused
  by reciprocal rank; keyword-only when no embeddings exist.
- Measured on this machine (Apple A18 Pro, 8 GB): qwen3 about 2.8 passages per
  second, nomic about 13. A 12-document session (820 passages) indexed in about
  3 minutes after the first document; searches take 0.1 to 0.25 seconds.

### Claims and evidence

- Analysis stage (`analysis.py`, `POST /research/{id}/analyze`): for each task
  in the plan, retrieve up to six passages (at most two per document), ask the
  model for up to five atomic claims, each with an exact quote and passage.
- A claim is kept only if its quote is found in the cited passage's stored text.
  This check is deterministic, not a model judgement.
- A second model call classifies each kept claim against its passage as
  supported, partially supported, unsupported, contradicted, or insufficient
  evidence, with a one-sentence reason.
- Evidence records carry the quote, document, section, passage, heading, and
  page. No confidence score is produced, because none would be calibrated.
- Passages are passed to the model as delimited untrusted text.
- Verified live: 28 claims in 57 seconds for a 7-task plan, about 39,000 tokens;
  18 supported, 9 partially supported, 1 unsupported; 1 claim dropped because
  its quote was not in the passage. The default token budget is now 200,000.

### Browser interface

- Responsive dashboard served by FastAPI at `http://127.0.0.1:8000`.
- Question form, optional search/source/token limits, and example questions.
- Saved-session sidebar, status summary, plan, source, evidence, timeline, and
  report tabs.
- Automatic refresh every two seconds while a session is searching.
- Continue discovery, refresh, JSON export, and confirmed session deletion.
- Read sources / retry unread sources, a per-source read status, and an
  expandable viewer for the extracted text with headings and page numbers.
- Passages tab with search; Analyze evidence button; claims shown with verdict,
  reason, quote, source link, heading, and page.
- Keyboard-accessible tabs, visible focus states, mobile layout, and safe text
  rendering for provider metadata.

### Development and verification

- Unit and integration-style tests for persistence, API validation, frontend
  assets, search parsing, budgeting, retries, provider failures, deduplication,
  continuation, deletion during work, and restart recovery.
- Tests for the fetcher's URL and redirect rules, both HTML extractors, PDF
  headings and pages, full-text location choice, the read flow through the API,
  the document budget, and restart recovery during reading.
- The latest local verification (2026-10-08) passed 46 pytest tests, Ruff checks,
  JavaScript syntax validation, and headless-Chrome checks of the reading,
  passage-search, and claims UI.
- Live read of a saved 10-source session: 8 read (4 arXiv PDFs of 11 to 36
  pages, 4 web articles); 2 failed with reported reasons (a `doi.org` target
  answering HTTP 202, and Medium answering HTTP 403).
- Local Docker Compose configuration for PostgreSQL and optional SearXNG, with
  SearXNG JSON results enabled.

## Current limitations

- Discovery metadata is not evidence. Titles, abstracts, and snippets must not
  be treated as verified support for claims.
- A verdict is the same model's second look at its own claim against one
  passage. It catches overstatement but is not independent confirmation, and a
  claim is never checked against other documents.
- No search for contradicting evidence, no claim deduplication, and no report.
- In the one live run, 22 of 28 claims came from two web pages. A per-document
  passage cap was added afterwards and has not been re-run live.
- The first document of the live indexing run took about 17 minutes against an
  expected 70 seconds; the cause was not established (memory pressure with two
  embedding models loaded is the leading guess). qwen3 holds about 2.4 GB while
  loaded.
- Vectors are compared in memory with NumPy, not pgvector. There is no reranker.
- Documents indexed before the 40-character minimum keep their short fragments.
- Reading limits: no JavaScript rendering, no OCR for scanned PDFs, no paywall
  or bot-challenge handling, and `robots.txt` is not consulted.
- PDF structure is heuristic: headings in body-size bold are missed, running
  headers and footers are kept as text, and tables come out as plain lines.
- If a landing page's advertised PDF cannot be fetched, the landing page itself
  is read, which may be only an abstract.
- The fetcher validates a host's addresses and then lets the HTTP client resolve
  it again, so DNS rebinding is not fully closed. Parsing runs in the API
  process, not a sandbox.
- There is no cross-session fetch cache; each session fetches its own copies.
- PyMuPDF is AGPL-licensed, which matters if ResearchOS is ever distributed or
  offered as a service.
- The model is used for planning and claim analysis, only when configured.
  Without a key the template plan is used and analysis is unavailable.
- Claims, evidence extraction, contradiction detection, citation auditing, and
  report generation are not implemented.
- The discovery worker is local and single-process. It is not a durable queue,
  does not support multiple Uvicorn workers or replicas, and only one session
  can search at a time.
- Progress polling is live only while the dashboard remains open. The SSE route
  replays saved events but is not a durable live event stream.
- Authentication, users, projects, roles, and collaboration are not implemented.
- PostgreSQL has not been tested in this workspace; all runs so far used SQLite.

## Provider status and configuration

| Provider | Current state | Required configuration |
| --- | --- | --- |
| arXiv | Confirmed the HTTP 406 is arXiv's export CDN intermittently rejecting well-formed requests, unrelated to query content, client library, HTTP version, or User-Agent (reproduced and cleared with plain curl, raw httpx, and the project adapter under identical conditions). 406 is now treated as a retryable, transient failure (`src/researchos/search.py`) instead of immediately suspending the provider for the run, so a live session keeps other providers running and picks arXiv back up on retry. No further root cause is available from the client side. | No API key. Some runs will still show a logged, non-fatal arXiv error if the CDN is degraded during that window; this is expected and does not stop discovery. |
| Semantic Scholar | Verified working end-to-end with a configured `SEMANTIC_SCHOLAR_API_KEY`. The shared/keyed rate-limit pool was tripping after only 1-2 requests because the adapter fired requests 1.1s apart; the request interval is now 3.2s (matching arXiv's pacing) and the 429 backoff without a `Retry-After` header rose from 5s to 20s. Also fixed: a 200 response with no `data` key is Semantic Scholar's genuine zero-hit shape (confirmed directly against the live API), not an error — it was previously misclassified as `"missing results array"` and could trip provider suspension. | `SEMANTIC_SCHOLAR_API_KEY` configured and confirmed working. Provider may still legitimately rate-limit and pause for a session under heavy use; this is expected, not a bug. |
| SearXNG | Verified working: started via `docker compose up -d searxng` using the committed `compose.yaml` and `config/searxng.yml`, confirmed JSON search results end-to-end. | `SEARXNG_URL=http://localhost:8080` with the container running. |

Supported `.env` entries at this stage:

```dotenv
DATABASE_URL=sqlite:///./researchos.db
DISCOVERY_ENABLED=true
SEARCH_TIMEOUT=30
SEMANTIC_SCHOLAR_API_KEY=
SEARXNG_URL=
```

Restart the API after changing `.env`. Existing sessions with an exhausted search
budget will need a new session; changing configuration does not restore spent
attempts.

## Remaining work

### 1. Stabilize source discovery

- Provider access is now stable: Semantic Scholar (keyed, re-paced) and SearXNG
  (local Docker container) both verified end-to-end; arXiv's intermittent 406
  is handled as a retryable transient failure rather than a hard stop; task
  status now accurately reflects per-task outcomes (see Completed work and
  Provider status table above).
- Add relevance ranking and better query generation before widening search volume.
- A provider is still suspended for the rest of a session after its retries are
  exhausted (arXiv or Semantic Scholar). This is a deliberate budget-conservation
  choice, not a bug, but it means a papers-only task can legitimately end up
  `failed` if both paper providers are down for that run. Two options if this
  proves too aggressive in practice: (a) let the planner avoid creating
  single-provider-dependent tasks, or (b) suspend per-task instead of
  session-wide so a later task gets a fresh attempt.
- Optional follow-up: since arXiv's 406 is CDN-side and outside client control,
  consider a small increase in per-provider retry attempts (independent of the
  global search-budget retry count) if it proves too flaky in practice.

### 2. Fetch and read documents

- Done: bounded fetching, HTML and PDF parsing with sections and pages, document
  hashes (see Completed work).
- Remaining: section-aware chunks (with step 3), a cross-session fetch cache,
  and the reading limits listed under Current limitations.

### 3. Add retrieval and model integration

- Done and verified live: provider choice (Ollama, `gpt-oss:120b`),
  question-specific planning, token accounting.
- Done: embeddings, vector storage, lexical retrieval, hybrid search.
- Remaining: metadata filters, reranking, pgvector if the corpus outgrows memory.

### 4. Build evidence and verification

- Done: atomic claims, exact quotes with section and page, support
  classification.
- Remaining: deliberate search for contradicting studies, cross-document
  checks, claim deduplication, and a citation audit before any report.

### 5. Produce research outputs

- Generate a source-grounded Markdown report with citations and uncertainty.
- Add report export and a clear distinction between discovery, evidence, and
  synthesized conclusions in the dashboard.
- Replace the local worker with a durable job queue and live progress stream.

### 6. Prepare for reliable use

- Add database migrations and PostgreSQL/pgvector integration tests.
- Add authentication and project-level ownership before multi-user deployment.
- Create a human-reviewed evaluation set and measure source quality, citation
  accuracy, coverage, cost, and latency.

## Recommended next action

Generate the report from supported claims (step 5), citing claim IDs so every
sentence traces to a quote. Before or alongside it: re-run analysis live to
confirm the per-document cap spreads claims across sources, and add relevance
ranking so off-topic papers stop filling the source budget.

For chronological implementation detail, see
[IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md).
