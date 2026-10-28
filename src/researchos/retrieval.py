"""Section-aware chunking and hybrid (semantic + keyword) passage search."""

import math
import re
from collections import Counter

import numpy as np

from researchos.embed import Embedder, EmbedFailure
from researchos.models import Chunk, Document, ResearchState, Section
from researchos.storage import Repository

TARGET_CHARS = 1500
OVERLAP_CHARS = 200
MIN_CHARS = 40
CANDIDATES = 50
RRF_K = 60
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_units(text: str) -> list[tuple[str, str]]:
    """(separator, unit) pairs: paragraphs, broken into sentences only when too long."""
    units = []
    for paragraph in text.split("\n\n"):
        if len(paragraph) <= TARGET_CHARS:
            units.append(("\n\n", paragraph))
            continue
        separator = "\n\n"
        for sentence in SENTENCE_END.split(paragraph):
            for start in range(0, len(sentence), TARGET_CHARS):
                units.append((separator, sentence[start : start + TARGET_CHARS]))
                separator = " "
    return [(separator, unit) for separator, unit in units if unit.strip()]


def chunk_sections(document: Document, sections: list[Section]) -> list[Chunk]:
    """Chunks never cross a section, so each keeps one heading and one page."""
    chunks = []
    for section in sections:
        current = ""
        last_unit = ""

        def close() -> None:
            # Fragments this short (stray table cells, figure labels) carry no evidence.
            if len(current.strip()) >= MIN_CHARS:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        section_id=section.id,
                        heading=section.heading,
                        page=section.page,
                        text=current.strip(),
                    )
                )

        for separator, unit in split_units(section.text):
            if current and len(current) + len(separator) + len(unit) > TARGET_CHARS:
                close()
                # A short trailing unit is repeated so a claim split across chunks stays whole.
                fits = len(last_unit) + len(separator) + len(unit) <= TARGET_CHARS
                current = last_unit if len(last_unit) <= OVERLAP_CHARS and fits else ""
            current = f"{current}{separator}{unit}" if current else unit
            last_unit = unit
        close()
    return chunks


def embedding_text(document: Document, chunk: Chunk) -> str:
    return "\n".join(part for part in (document.title, chunk.heading, chunk.text) if part)


def index_document(
    repository: Repository,
    embedder: Embedder | None,
    model: str | None,
    state: ResearchState,
    document: Document,
) -> None:
    """Chunk a document and, when a model is available, embed the chunks."""
    chunks = chunk_sections(document, repository.get_sections(state.id, document.id))
    vectors = None
    if embedder and model:
        vectors = embedder.embed(model, [embedding_text(document, chunk) for chunk in chunks])
    repository.replace_chunks(state.id, document.id, chunks, vectors)
    document.chunk_count = len(chunks)
    document.embedding_model = model if vectors is not None else None


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_ranking(query: str, texts: list[str]) -> list[int]:
    """BM25 over the session's chunks; indexes of matching chunks, best first."""
    documents = [Counter(tokens(text)) for text in texts]
    lengths = [sum(counts.values()) for counts in documents]
    average = (sum(lengths) / len(lengths)) or 1
    scores = [0.0] * len(texts)
    for term in set(tokens(query)):
        containing = sum(1 for counts in documents if term in counts)
        if not containing:
            continue
        idf = math.log(1 + (len(texts) - containing + 0.5) / (containing + 0.5))
        for index, counts in enumerate(documents):
            frequency = counts[term]
            if frequency:
                norm = frequency + 1.5 * (0.25 + 0.75 * lengths[index] / average)
                scores[index] += idf * frequency * 2.5 / norm
    ranked = sorted(range(len(texts)), key=lambda index: scores[index], reverse=True)
    return [index for index in ranked if scores[index] > 0][:CANDIDATES]


def search(
    repository: Repository,
    embedder: Embedder | None,
    state: ResearchState,
    query: str,
    limit: int = 8,
    document_ids: set[str] | None = None,
) -> dict:
    """Fuse semantic and keyword rankings; fall back to keywords if embedding fails."""
    chunks, matrix = repository.get_chunks(state.id)
    if document_ids is not None:
        keep = [index for index, chunk in enumerate(chunks) if chunk.document_id in document_ids]
        chunks = [chunks[index] for index in keep]
        matrix = matrix[keep] if matrix is not None else None
    if not chunks:
        return {"mode": "none", "model": None, "results": []}
    rankings = {"keyword": keyword_ranking(query, [chunk.text for chunk in chunks])}
    mode = "keyword"
    if embedder and state.embedding_model and matrix is not None:
        try:
            vector = embedder.embed(state.embedding_model, [query], query=True)[0]
            similarity = matrix @ vector
            rankings["semantic"] = list(np.argsort(-similarity)[:CANDIDATES])
            mode = "hybrid"
        except EmbedFailure:
            pass
    fused: Counter = Counter()
    for ranking in rankings.values():
        for rank, index in enumerate(ranking):
            fused[int(index)] += 1 / (RRF_K + rank + 1)
    documents = {document.id: document for document in state.documents}
    sources = {source.id: source for source in state.sources}
    results = []
    for index, score in fused.most_common(limit):
        chunk = chunks[index]
        document = documents.get(chunk.document_id)
        source = sources.get(document.source_id) if document else None
        results.append(
            {
                "chunk": chunk,
                "score": round(score, 5),
                "source_id": source.id if source else None,
                "source_title": source.title if source else (document.title if document else ""),
                "source_url": str(source.url) if source else None,
            }
        )
    return {
        "mode": mode,
        "model": state.embedding_model if mode == "hybrid" else None,
        "results": results,
    }
