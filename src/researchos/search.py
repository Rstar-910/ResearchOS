"""Read-only discovery adapters. Search metadata is not verified evidence."""

import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from researchos.config import Settings
from researchos.models import Source


class SearchFailure(Exception):
    def __init__(
        self,
        message: str,
        retryable: bool = False,
        retry_after: float = 0,
        suspend_provider: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.suspend_provider = suspend_provider


def retry_delay(value: str | None) -> float:
    if value is None:
        # Semantic Scholar's shared rate-limit pool sends no Retry-After header;
        # a short backoff just re-triggers the same 429.
        return 20
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 5


def failure_hint(provider: str, code: int) -> str:
    if code == 429:
        if provider == "semantic_scholar":
            return "Rate limited. Wait, then retry; configure SEMANTIC_SCHOLAR_API_KEY if absent."
        return "Rate limited. Wait before starting another run."
    if code in {401, 403} and provider == "semantic_scholar":
        return "Access rejected. Check your Semantic Scholar API key and access permissions."
    if code == 406 and provider == "arxiv":
        return (
            "arXiv's export CDN intermittently rejects well-formed requests with 406 "
            "regardless of query content; this is usually transient and is retried "
            "automatically."
        )
    if code == 403 and provider == "arxiv":
        return (
            "Request rejected. No arXiv key is required; run the provider diagnostic "
            "and check arXiv service/access status."
        )
    if code == 403 and provider == "searxng":
        return (
            "Access rejected. Check the instance permissions "
            "and enable search.formats: [html, json]."
        )
    return "Provider request failed; inspect service availability before retrying."


class SearchProvider(Protocol):
    name: str
    source_type: str

    def search(self, query: str, limit: int) -> list[Source]: ...


class HTTPProvider:
    def __init__(self, client: httpx.Client, name: str, source_type: str, interval: float):
        self.client, self.name, self.source_type = client, name, source_type
        self.interval = interval
        self.lock = threading.Lock()
        self.last_call = 0.0

    def fetch(self, url: str, **kwargs) -> bytes:
        with self.lock:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            try:
                with self.client.stream("GET", url, **kwargs) as response:
                    if response.status_code >= 400:
                        code = response.status_code
                        delay = retry_delay(response.headers.get("Retry-After"))
                        # Do not shorten a provider's requested cooldown. Long cooldowns
                        # stop this provider for the run instead of blocking the worker.
                        # arXiv's export CDN intermittently returns 406 for well-formed
                        # requests regardless of query content; treat it as transient.
                        retryable = (code == 429 or code >= 500 or code == 406) and delay <= 60
                        raise SearchFailure(
                            f"{self.name}: HTTP {code}. {failure_hint(self.name, code)}",
                            retryable,
                            delay,
                            suspend_provider=code in {401, 403, 406, 429} or delay > 60,
                        )
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > 5_000_000:
                            raise SearchFailure(f"{self.name}: response exceeds 5 MB")
                    return bytes(data)
            except httpx.HTTPError:
                # Never expose headers, secrets, or raw provider responses in user errors.
                raise SearchFailure(
                    f"{self.name}: connection or timeout error. "
                    "Check internet, DNS, and proxy settings.",
                    True,
                ) from None


class Arxiv(HTTPProvider):
    def __init__(self, client):
        super().__init__(client, "arxiv", "papers", 3.1)

    def search(self, query: str, limit: int) -> list[Source]:
        terms = re.findall(r"[\w-]+", query)[:40]
        body = self.fetch(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": " AND ".join(f"all:{term}" for term in terms),
                "start": 0,
                "max_results": limit,
            },
        )
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        if b"<!DOCTYPE" in body or b"<!ENTITY" in body:
            raise SearchFailure("arxiv: unexpected XML declaration")
        try:
            entries = ET.fromstring(body).findall("a:entry", ns)
            results = []
            for entry in entries:
                url = entry.findtext("a:id", "", ns)
                if "/api/errors" in url:
                    raise SearchFailure("arxiv: query rejected")
                try:
                    results.append(
                        Source(
                            title=" ".join(entry.findtext("a:title", "", ns).split()),
                            url=url,
                            provider=self.name,
                            source_type="paper",
                            snippet=entry.findtext("a:summary", "", ns).strip(),
                            authors=[
                                a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)
                            ],
                            published_at=entry.findtext("a:published", None, ns),
                            doi=entry.findtext("x:doi", None, ns),
                            arxiv_id=url.split("/abs/")[-1],
                        )
                    )
                except ValidationError:
                    continue
            return results
        except ET.ParseError:
            raise SearchFailure("arxiv: invalid XML response") from None


class JSONProvider(HTTPProvider):
    def json(self, url: str, **kwargs) -> dict:
        import json

        try:
            result = json.loads(self.fetch(url, **kwargs))
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (ValueError, UnicodeError):
            raise SearchFailure(f"{self.name}: invalid JSON response") from None


class SemanticScholar(JSONProvider):
    def __init__(self, client, key: str | None = None):
        super().__init__(client, "semantic_scholar", "papers", 3.2)
        self.key = key

    def search(self, query: str, limit: int) -> list[Source]:
        data = self.json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,url,abstract,authors,year,externalIds,openAccessPdf",
            },
            headers={"x-api-key": self.key} if self.key else {},
        )
        # A 200 response can omit "data" entirely for a zero-hit search; treat
        # a missing key as no results rather than a hard provider failure.
        if "data" not in data:
            return []
        if not isinstance(data["data"], list):
            raise SearchFailure("semantic_scholar: missing results array")
        results = []
        for item in data["data"][:limit]:
            try:
                ids = item.get("externalIds") or {}
                results.append(
                    Source(
                        title=item["title"],
                        url=item["url"],
                        provider=self.name,
                        source_type="paper",
                        snippet=item.get("abstract") or "",
                        authors=[a["name"] for a in item.get("authors", [])],
                        published_at=str(item["year"]) if item.get("year") else None,
                        doi=ids.get("DOI"),
                        arxiv_id=ids.get("ArXiv"),
                        pdf_url=(item.get("openAccessPdf") or {}).get("url") or None,
                    )
                )
            except (KeyError, TypeError, AttributeError, ValidationError):
                continue
        return results


class Searxng(JSONProvider):
    def __init__(self, client, url: str):
        super().__init__(client, "searxng", "web", 0.5)
        self.url = url.rstrip("/") + "/search"

    def search(self, query: str, limit: int) -> list[Source]:
        data = self.json(self.url, params={"q": query, "format": "json"})
        if not isinstance(data.get("results"), list):
            raise SearchFailure("searxng: missing results array; enable JSON search format")
        results = []
        for item in data["results"][:limit]:
            try:
                results.append(
                    Source(
                        title=item["title"],
                        url=item["url"],
                        provider=self.name,
                        source_type="web",
                        snippet=item.get("content") or "",
                    )
                )
            except (KeyError, TypeError, ValidationError):
                continue
        return results


class ToolRegistry:
    def __init__(self, providers: list[SearchProvider]):
        self.providers = providers

    def for_types(self, types: list[str]) -> list[SearchProvider]:
        return [provider for provider in self.providers if provider.source_type in types]


def configured_registry(settings: Settings, client: httpx.Client) -> ToolRegistry:
    if not settings.discovery_enabled:
        return ToolRegistry([])
    key = settings.semantic_scholar_api_key
    providers = [Arxiv(client), SemanticScholar(client, key.get_secret_value() if key else None)]
    if settings.searxng_url:
        providers.append(Searxng(client, str(settings.searxng_url)))
    return ToolRegistry(providers)


def identity_keys(source: Source) -> set[str]:
    keys = set()
    if source.doi:
        keys.add("doi:" + re.sub(r"^https?://(?:dx\.)?doi.org/", "", source.doi.lower()))
    if source.arxiv_id:
        keys.add("arxiv:" + re.sub(r"v\d+$", "", source.arxiv_id))
    for url in [str(source.url), *source.aliases]:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host in {"arxiv.org", "export.arxiv.org"}:
            paper = re.sub(r"^/(abs|pdf)/", "", parts.path).removesuffix(".pdf")
            keys.add("arxiv:" + re.sub(r"v\d+$", "", paper))
        if host in {"doi.org", "dx.doi.org"}:
            keys.add("doi:" + parts.path.lstrip("/").lower())
        query = urlencode(
            sorted(
                (k, v)
                for k, v in parse_qsl(parts.query)
                if not k.startswith("utm_") and k not in {"fbclid", "gclid"}
            )
        )
        keys.add(
            "url:"
            + urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), query, ""))
        )
    return keys


def merge_sources(existing: list[Source], incoming: list[Source], limit: int) -> None:
    for source in incoming:
        source.providers = list(dict.fromkeys([source.provider, *source.providers]))
        matches = [item for item in existing if identity_keys(item) & identity_keys(source)]
        if not matches:
            if len(existing) < limit:
                existing.append(source)
            continue
        target = matches[0]
        for item in [source, *matches[1:]]:
            target.providers = list(
                dict.fromkeys([target.provider, *target.providers, item.provider, *item.providers])
            )
            target.aliases = list(dict.fromkeys([*target.aliases, str(item.url), *item.aliases]))
            target.doi = target.doi or item.doi
            target.arxiv_id = target.arxiv_id or item.arxiv_id
            target.pdf_url = target.pdf_url or item.pdf_url
            target.snippet = target.snippet or item.snippet
            target.authors = target.authors or item.authors
            target.published_at = target.published_at or item.published_at
        for item in matches[1:]:
            existing.remove(item)
