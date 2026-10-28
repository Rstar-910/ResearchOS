import numpy as np
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    select,
    update,
)

from researchos.models import Chunk, ResearchState, Section

BUSY = ("searching", "reading", "analyzing")


class Repository:
    """Session snapshots; no shared connection or process-local session cache."""

    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True)
        self.metadata = MetaData()
        self.sessions = Table(
            "research_sessions",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("state", JSON, nullable=False),
        )
        # Section text lives outside the snapshot so polling a session stays cheap.
        self.sections = Table(
            "document_sections",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("session_id", String(36), nullable=False, index=True),
            Column("document_id", String(36), nullable=False, index=True),
            Column("position", Integer, nullable=False),
            Column("heading", Text, nullable=False),
            Column("page", Integer),
            Column("text", Text, nullable=False),
        )

        self.chunks = Table(
            "document_chunks",
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("session_id", String(36), nullable=False, index=True),
            Column("document_id", String(36), nullable=False, index=True),
            Column("section_id", String(36), nullable=False),
            Column("position", Integer, nullable=False),
            Column("heading", Text, nullable=False),
            Column("page", Integer),
            Column("text", Text, nullable=False),
            Column("embedding", LargeBinary),
        )

    def initialize(self) -> None:
        self.metadata.create_all(self.engine)

    def create(self, state: ResearchState) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                self.sessions.insert().values(
                    id=state.id,
                    state=state.model_dump(mode="json"),
                )
            )

    def save(self, state: ResearchState) -> bool:
        with self.engine.begin() as connection:
            return (
                connection.execute(
                    update(self.sessions)
                    .where(self.sessions.c.id == state.id)
                    .values(state=state.model_dump(mode="json"))
                ).rowcount
                == 1
            )

    def claim_search(self, state: ResearchState) -> bool:
        """Save a state that starts background work, unless the session is already busy."""
        with self.engine.begin() as connection:
            return (
                connection.execute(
                    update(self.sessions)
                    .where(
                        self.sessions.c.id == state.id,
                        self.sessions.c.state["status"].as_string().not_in(BUSY),
                    )
                    .values(state=state.model_dump(mode="json"))
                ).rowcount
                == 1
            )

    def add_sections(self, session_id: str, document_id: str, sections: list[Section]) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                self.sections.insert(),
                [
                    {
                        "id": section.id,
                        "session_id": session_id,
                        "document_id": document_id,
                        "position": position,
                        "heading": section.heading,
                        "page": section.page,
                        "text": section.text,
                    }
                    for position, section in enumerate(sections)
                ],
            )

    def get_sections(self, session_id: str, document_id: str) -> list[Section]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(self.sections)
                .where(
                    self.sections.c.session_id == session_id,
                    self.sections.c.document_id == document_id,
                )
                .order_by(self.sections.c.position)
            ).mappings()
            return [
                Section(id=row["id"], heading=row["heading"], text=row["text"], page=row["page"])
                for row in rows
            ]

    def delete_sections(self, session_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                delete(self.sections).where(self.sections.c.session_id == session_id)
            )
            connection.execute(delete(self.chunks).where(self.chunks.c.session_id == session_id))

    def replace_chunks(
        self, session_id: str, document_id: str, chunks: list[Chunk], vectors: np.ndarray | None
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(delete(self.chunks).where(self.chunks.c.document_id == document_id))
            if chunks:
                connection.execute(
                    self.chunks.insert(),
                    [
                        {
                            "id": chunk.id,
                            "session_id": session_id,
                            "document_id": document_id,
                            "section_id": chunk.section_id,
                            "position": position,
                            "heading": chunk.heading,
                            "page": chunk.page,
                            "text": chunk.text,
                            "embedding": None
                            if vectors is None
                            else vectors[position].astype(np.float32).tobytes(),
                        }
                        for position, chunk in enumerate(chunks)
                    ],
                )

    def get_chunks(self, session_id: str) -> tuple[list[Chunk], np.ndarray | None]:
        """All chunks of a session, and their vectors when every chunk has one."""
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(self.chunks)
                    .where(self.chunks.c.session_id == session_id)
                    .order_by(self.chunks.c.document_id, self.chunks.c.position)
                )
                .mappings()
                .all()
            )
        chunks = [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                section_id=row["section_id"],
                heading=row["heading"],
                page=row["page"],
                text=row["text"],
            )
            for row in rows
        ]
        blobs = [row["embedding"] for row in rows]
        sizes = {len(blob) for blob in blobs if blob}
        if not rows or not all(blobs) or len(sizes) != 1:
            return chunks, None
        return chunks, np.vstack([np.frombuffer(blob, dtype=np.float32) for blob in blobs])

    def recover_interrupted(self) -> None:
        with self.engine.connect() as connection:
            snapshots = (
                connection.execute(
                    select(self.sessions.c.state).where(
                        self.sessions.c.state["status"].as_string().in_(BUSY)
                    )
                )
                .scalars()
                .all()
            )
        for snapshot in snapshots:
            state = ResearchState.model_validate(snapshot)
            if state.status == "reading":
                state.status = "reading_complete" if state.documents else "discovery_complete"
                state.record(
                    "interrupted", "Server restarted while reading. Read sources again to resume."
                )
                self.save(state)
                continue
            if state.status == "analyzing":
                state.status = "analysis_complete" if state.claims else "reading_complete"
                state.record(
                    "interrupted", "Server restarted during analysis. Analyze again to resume."
                )
                self.save(state)
                continue
            state.status = "interrupted"
            for event in state.search_history:
                if event.status == "running":
                    event.status = "failed"
                    event.error = "Interrupted by server restart; attempt remains charged."
            state.record(
                "interrupted", "Server restarted. Continue discovery to retry remaining searches."
            )
            self.save(state)

    def get(self, research_id: str) -> ResearchState | None:
        with self.engine.connect() as connection:
            data = connection.execute(
                select(self.sessions.c.state).where(self.sessions.c.id == research_id)
            ).scalar_one_or_none()
        return ResearchState.model_validate(data) if data is not None else None

    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        state = self.sessions.c.state
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    select(
                        self.sessions.c.id,
                        state["question"].as_string().label("question"),
                        state["status"].as_string().label("status"),
                        state["updated_at"].as_string().label("updated_at"),
                    )
                    .order_by(state["updated_at"].as_string().desc(), self.sessions.c.id)
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def delete(self, research_id: str) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(self.sessions).where(self.sessions.c.id == research_id)
            )
            connection.execute(
                delete(self.sections).where(self.sections.c.session_id == research_id)
            )
            connection.execute(delete(self.chunks).where(self.chunks.c.session_id == research_id))
            return result.rowcount == 1
