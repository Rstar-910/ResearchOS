from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def new_id() -> str:
    return str(uuid4())


def now() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Budget(Model):
    max_searches: int = Field(default=20, ge=1, le=100)
    max_sources: int = Field(default=20, ge=1, le=100)
    max_tokens: int = Field(default=200000, ge=1, le=1000000)
    max_documents: int = Field(default=20, ge=1, le=100)


class ResearchRequest(Model):
    question: str = Field(min_length=3, max_length=4000)
    budget: Budget = Field(default_factory=Budget)

    @field_validator("question", mode="before")
    @classmethod
    def normalize(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value


class Task(Model):
    id: str = Field(default_factory=new_id)
    question: str
    source_types: list[str]
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    query: str | None = None
    analysis: Literal["pending", "completed", "failed", "skipped"] = "pending"


class Source(Model):
    id: str = Field(default_factory=new_id)
    title: str
    url: HttpUrl
    provider: str
    source_type: str
    retrieved_at: datetime = Field(default_factory=now)
    snippet: str = ""
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    providers: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class SearchEvent(Model):
    provider: str
    query: str
    task_id: str
    status: Literal["running", "completed", "failed"] = "running"
    results_count: int = 0
    error: str | None = None
    timestamp: datetime = Field(default_factory=now)


class Section(Model):
    id: str = Field(default_factory=new_id)
    heading: str
    text: str
    page: int | None = Field(default=None, ge=1)


class Chunk(Model):
    """A passage of one section: the unit that is embedded, searched, and cited."""

    id: str = Field(default_factory=new_id)
    document_id: str
    section_id: str
    heading: str = ""
    page: int | None = None
    text: str


class Document(Model):
    """Document metadata. Section text is stored separately, outside the session snapshot."""

    id: str = Field(default_factory=new_id)
    source_id: str
    content_hash: str
    url: str = ""
    title: str = ""
    content_type: Literal["html", "pdf", "text"] = "html"
    pages: int | None = None
    section_count: int = 0
    char_count: int = 0
    truncated: bool = False
    chunk_count: int = 0
    embedding_model: str | None = None
    retrieved_at: datetime = Field(default_factory=now)


class ReadEvent(Model):
    source_id: str
    status: Literal["completed", "duplicate", "failed"]
    url: str | None = None
    document_id: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=now)


class Evidence(Model):
    id: str = Field(default_factory=new_id)
    claim_id: str
    document_id: str
    section_id: str
    chunk_id: str | None = None
    heading: str = ""
    text: str
    page: int | None = Field(default=None, ge=1)
    evidence_type: Literal["direct", "indirect", "inferred", "contextual", "contradictory"]
    # No calibrated confidence exists yet, so none is invented.
    confidence: float | None = Field(default=None, ge=0, le=1)
    extracted_at: datetime = Field(default_factory=now)


class Claim(Model):
    id: str = Field(default_factory=new_id)
    task_id: str | None = None
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    verification: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE",
    ] = "INSUFFICIENT_EVIDENCE"
    verification_reason: str = "Not yet verified."


class Event(Model):
    sequence: int
    action: str
    message: str
    timestamp: datetime = Field(default_factory=now)


class ResearchState(Model):
    id: str = Field(default_factory=new_id)
    question: str
    status: Literal[
        "created",
        "planning",
        "awaiting_providers",
        "searching",
        "discovery_complete",
        "discovery_failed",
        "interrupted",
        "reading",
        "reading_complete",
        "reading_failed",
        "analyzing",
        "analysis_complete",
        "analysis_failed",
        "completed",
        "failed",
    ] = "created"
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    budget: Budget = Field(default_factory=Budget)
    tasks: list[Task] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    token_usage: int = 0
    report: str | None = None
    search_history: list[SearchEvent] = Field(default_factory=list)
    read_history: list[ReadEvent] = Field(default_factory=list)
    plan_source: Literal["template", "model"] = "template"
    embedding_model: str | None = None

    def record(self, action: str, message: str) -> None:
        self.updated_at = now()
        self.events.append(Event(sequence=len(self.events) + 1, action=action, message=message))
