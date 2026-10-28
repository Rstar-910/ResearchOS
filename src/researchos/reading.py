import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import urljoin, urlsplit

from researchos.embed import Embedder, EmbedFailure
from researchos.fetch import Fetched, Fetcher, FetchFailure
from researchos.models import Document, ReadEvent, ResearchState, Source
from researchos.parse import Parsed, ParseFailure, parse, pdf_link
from researchos.retrieval import index_document
from researchos.storage import Repository

READABLE = {"discovery_complete", "reading_complete", "reading_failed"}
ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
# Index pages that describe a paper without containing it; reading them would pass
# an abstract off as the document.
METADATA_HOSTS = {"semanticscholar.org", "www.semanticscholar.org", "api.semanticscholar.org"}
MAX_CANDIDATES = 3


def candidate_urls(source: Source) -> list[str]:
    """Full-text locations for a source, best first."""
    urls = [source.pdf_url] if source.pdf_url else []
    pages = []
    arxiv_id = source.arxiv_id
    for url in [str(source.url), *source.aliases]:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host in ARXIV_HOSTS:
            arxiv_id = arxiv_id or re.sub(r"^/(abs|pdf)/", "", parts.path).removesuffix(".pdf")
        elif host not in METADATA_HOSTS:
            pages.append(url)
    if arxiv_id and ARXIV_ID.match(arxiv_id):
        urls.append(f"https://arxiv.org/pdf/{arxiv_id}")
    urls.extend(pages)
    if source.doi:
        urls.append("https://doi.org/" + re.sub(r"^https?://(?:dx\.)?doi\.org/", "", source.doi))
    return list(dict.fromkeys(urls))[:MAX_CANDIDATES]


def unread_sources(state: ResearchState) -> list[Source]:
    done = {e.source_id for e in state.read_history if e.status in {"completed", "duplicate"}}
    return [source for source in state.sources if source.id not in done]


def unindexed(state: ResearchState, embedder: Embedder | None) -> list[Document]:
    """Documents with no passages yet, or no embeddings although a model is configured."""
    return [
        document
        for document in state.documents
        if not document.chunk_count or (embedder and not document.embedding_model)
    ]


class Reader:
    """Single-process worker that fetches, parses, and indexes discovered sources."""

    def __init__(self, repository: Repository, fetcher: Fetcher, embedder: Embedder | None = None):
        self.repository, self.fetcher, self.embedder = repository, fetcher, embedder
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reading")
        self.lock = Lock()
        self.active = False

    def start(self, state: ResearchState) -> ResearchState:
        with self.lock:
            if self.active:
                raise ValueError("Reading is busy. Wait for the current session to finish.")
            if state.status == "reading":
                raise ValueError("This session is already reading sources.")
            if state.status not in READABLE:
                raise ValueError("Sources can be read once discovery is complete.")
            exhausted = len(state.documents) >= state.budget.max_documents
            if not unindexed(state, self.embedder):
                if exhausted:
                    raise ValueError("The session document budget is exhausted.")
                if not unread_sources(state):
                    raise ValueError("Every discovered source has already been read.")
            state.status = "reading"
            state.record("reading", "Reading discovered sources. Document text is untrusted data.")
            if not self.repository.claim_search(state):
                raise ValueError("Session unavailable or already busy.")
            self.active = True
            self.executor.submit(self.run, state.model_copy(deep=True))
            return state

    def fetch_full_text(self, url: str) -> tuple[Fetched, Parsed]:
        fetched = self.fetcher.fetch(url)
        link = pdf_link(fetched.body) if fetched.kind == "html" else None
        if link:
            # A scholarly landing page holds only the abstract; prefer the paper it points to.
            try:
                paper = self.fetcher.fetch(urljoin(fetched.url, link))
                if paper.kind == "pdf":
                    return paper, parse(paper)
            except (FetchFailure, ParseFailure):
                pass
        return fetched, parse(fetched)

    def read(self, state: ResearchState, source: Source) -> ReadEvent:
        urls = candidate_urls(source)
        if not urls:
            return ReadEvent(
                source_id=source.id,
                status="failed",
                error="No full-text location is known; only search metadata is available.",
            )
        failures = []
        for url in urls:
            try:
                fetched, parsed = self.fetch_full_text(url)
            except (FetchFailure, ParseFailure) as error:
                failures.append(f"{urlsplit(url).hostname}: {error}")
                continue
            content_hash = hashlib.sha256(fetched.body).hexdigest()
            for existing in state.documents:
                if existing.content_hash == content_hash:
                    return ReadEvent(
                        source_id=source.id,
                        status="duplicate",
                        url=fetched.url,
                        document_id=existing.id,
                    )
            document = Document(
                source_id=source.id,
                content_hash=content_hash,
                url=fetched.url,
                title=parsed.title or source.title,
                content_type=fetched.kind,
                pages=parsed.pages,
                section_count=len(parsed.sections),
                char_count=sum(len(section.text) for section in parsed.sections),
                truncated=parsed.truncated,
            )
            self.repository.add_sections(state.id, document.id, parsed.sections)
            state.documents.append(document)
            return ReadEvent(
                source_id=source.id, status="completed", url=fetched.url, document_id=document.id
            )
        return ReadEvent(
            source_id=source.id, status="failed", url=urls[-1], error="; ".join(failures)
        )

    def index(self, state: ResearchState) -> bool:
        """Split read documents into passages and embed them. False if the session is gone."""
        pending = unindexed(state, self.embedder)
        if not pending:
            return True
        model = None
        if self.embedder:
            try:
                model = state.embedding_model or self.embedder.choose()
                state.embedding_model = model
            except EmbedFailure as error:
                state.record(
                    "index_error",
                    f"Embeddings unavailable ({error}); passages are searchable by keyword only.",
                )
        state.record(
            "indexing",
            f"Indexing {len(pending)} documents"
            + (f" with {model}." if model else " for keyword search."),
        )
        if not self.repository.save(state):
            return False
        for document in pending:
            if document.chunk_count and not model:
                continue
            try:
                index_document(self.repository, self.embedder, model, state, document)
            except EmbedFailure as error:
                state.record("index_error", f"Embedding stopped ({error}). Retry to finish.")
                model = None
                if not document.chunk_count:
                    index_document(self.repository, None, None, state, document)
            if not self.repository.save(state):
                self.repository.delete_sections(state.id)
                return False
        passages = sum(document.chunk_count for document in state.documents)
        embedded = sum(1 for document in state.documents if document.embedding_model)
        state.record(
            "indexed",
            f"{passages} passages indexed; "
            f"{embedded} of {len(state.documents)} documents embedded.",
        )
        return True

    def run(self, state: ResearchState) -> None:
        try:
            for source in unread_sources(state):
                if len(state.documents) >= state.budget.max_documents:
                    break
                event = self.read(state, source)
                state.read_history.append(event)
                if event.status == "completed":
                    document = state.documents[-1]
                    pages = f", {document.pages} pages" if document.pages else ""
                    state.record(
                        "document_read",
                        f"{source.title}: {document.section_count} sections{pages} "
                        f"({document.content_type}).",
                    )
                elif event.status == "duplicate":
                    state.record(
                        "document_duplicate",
                        f"{source.title}: same content as a document already read.",
                    )
                else:
                    message = f"Could not read {source.title}: {event.error}"
                    state.errors.append(message)
                    state.record("read_error", message)
                if not self.repository.save(state):
                    # The session was deleted while reading.
                    self.repository.delete_sections(state.id)
                    return
            if not self.index(state):
                return
            state.status = "reading_complete" if state.documents else "reading_failed"
            unread = len(unread_sources(state))
            state.record(
                state.status,
                f"{len(state.documents)} documents read; {unread} sources unread. "
                "Claims and reports are not generated yet.",
            )
            self.repository.save(state)
        except Exception:
            state.status = "reading_complete" if state.documents else "reading_failed"
            state.record("reading_failed", "Unexpected reading failure. Read documents are kept.")
            self.repository.save(state)
        finally:
            with self.lock:
                self.active = False

    def close(self) -> None:
        self.executor.shutdown(wait=True)
