"""Text embeddings from a local Ollama server."""

import httpx
import numpy as np

from researchos.config import Settings

# (document prefix, query prefix) each model was trained with.
PREFIXES = {
    "qwen3-embedding": (
        "",
        "Instruct: Given a research question, retrieve passages that answer it\nQuery: ",
    ),
    "nomic-embed-text": ("search_document: ", "search_query: "),
}
BATCH = 16


class EmbedFailure(Exception):
    pass


class Embedder:
    def __init__(
        self, client: httpx.Client, base_url: str, models: list[str], timeout: float = 300
    ):
        self.client, self.models, self.timeout = client, models, timeout
        self.url = base_url.rstrip("/") + "/api/embed"

    def embed(self, model: str, texts: list[str], query: bool = False) -> np.ndarray:
        """Unit-length vectors, one row per text."""
        prefix = PREFIXES.get(model.split(":")[0], ("", ""))[1 if query else 0]
        rows = []
        for start in range(0, len(texts), BATCH):
            batch = [prefix + text for text in texts[start : start + BATCH]]
            try:
                response = self.client.post(
                    self.url, timeout=self.timeout, json={"model": model, "input": batch}
                )
            except httpx.HTTPError:
                raise EmbedFailure("embedding server is not reachable") from None
            if response.status_code != 200:
                raise EmbedFailure(f"embedding model {model} failed: HTTP {response.status_code}")
            try:
                vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
            except (ValueError, KeyError, TypeError):
                raise EmbedFailure("embedding server returned an unreadable response") from None
            if vectors.ndim != 2 or len(vectors) != len(batch):
                raise EmbedFailure("embedding server returned the wrong number of vectors")
            rows.append(vectors)
        matrix = np.vstack(rows)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    def choose(self) -> str:
        """The first configured model that answers; vectors from different models never mix."""
        failure = "no embedding model is configured"
        for model in self.models:
            try:
                self.embed(model, ["ready"])
                return model
            except EmbedFailure as error:
                failure = str(error)
        raise EmbedFailure(failure)


def configured_embedder(settings: Settings, client: httpx.Client) -> Embedder | None:
    if not settings.embedding_base_url:
        return None
    models = [settings.embedding_model, settings.embedding_fallback_model]
    return Embedder(client, str(settings.embedding_base_url), [m for m in models if m])
