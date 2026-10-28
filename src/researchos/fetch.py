"""Bounded, read-only document fetching. Fetched content is untrusted data."""

import ipaddress
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = 20_000_000
MAX_REDIRECTS = 5
DEADLINE_SECONDS = 90
ACCEPT = "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8"
HOST_INTERVALS = {"arxiv.org": 3.1, "export.arxiv.org": 3.1}


class FetchFailure(Exception):
    pass


@dataclass(frozen=True)
class Fetched:
    url: str
    kind: Literal["html", "pdf", "text"]
    body: bytes
    charset: str | None = None


def resolve_host(host: str) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def classify(content_type: str, body: bytes) -> Literal["html", "pdf", "text"]:
    declared = content_type.split(";")[0].strip().lower()
    head = body[:1024].lstrip()
    if head.startswith(b"%PDF-"):
        return "pdf"
    if declared == "application/pdf":
        raise FetchFailure("response is labelled PDF but is not a PDF")
    if declared in {"text/html", "application/xhtml+xml"}:
        return "html"
    if declared == "text/plain":
        return "text"
    if not declared and head[:200].lower().startswith((b"<!doctype html", b"<html")):
        return "html"
    raise FetchFailure(f"unsupported content type: {declared[:60] or 'unknown'}")


class Fetcher:
    def __init__(
        self,
        client: httpx.Client,
        resolve: Callable[[str], list[str]] = resolve_host,
        default_interval: float = 1.0,
    ):
        self.client, self.resolve, self.default_interval = client, resolve, default_interval
        self.last_call: dict[str, float] = {}

    def check(self, url: str) -> str:
        """Reject anything that is not a plain public web address; return its host."""
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            raise FetchFailure("invalid URL") from None
        host = (parts.hostname or "").lower()
        if parts.scheme not in {"http", "https"} or not host:
            raise FetchFailure("unsupported URL")
        if parts.username or parts.password or port not in {None, 80, 443}:
            raise FetchFailure("URL uses credentials or a non-standard port")
        try:
            addresses = self.resolve(host)
        except OSError:
            raise FetchFailure("host could not be resolved") from None
        if not addresses or not all(is_public(address) for address in addresses):
            raise FetchFailure("host does not resolve to a public address")
        return host

    def wait(self, host: str) -> None:
        interval = HOST_INTERVALS.get(host, self.default_interval) if self.default_interval else 0
        elapsed = time.monotonic() - self.last_call.get(host, float("-inf"))
        time.sleep(max(0, interval - elapsed))
        self.last_call[host] = time.monotonic()

    def fetch(self, url: str) -> Fetched:
        deadline = time.monotonic() + DEADLINE_SECONDS
        requested = None
        for _ in range(MAX_REDIRECTS + 1):
            # Every hop is validated, so a public page cannot redirect to a private address.
            self.wait(self.check(url))
            requested = requested or urlsplit(url)
            try:
                with self.client.stream(
                    "GET", url, follow_redirects=False, headers={"Accept": ACCEPT}
                ) as response:
                    code = response.status_code
                    if code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchFailure("redirect without a location")
                        url = urljoin(url, location)
                        target = urlsplit(url)
                        parent = target.path.rstrip("/") + "/"
                        # Sites often answer a removed page by redirecting to an index
                        # page; reading that would attribute the wrong text to the source.
                        if (
                            target.hostname == requested.hostname
                            and requested.path.startswith(parent)
                            and requested.path.rstrip("/") != target.path.rstrip("/")
                        ):
                            raise FetchFailure("redirected to an index page; the page is gone")
                        continue
                    if code != 200:
                        raise FetchFailure(f"HTTP {code}")
                    length = response.headers.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_BYTES:
                        raise FetchFailure("document exceeds the 20 MB limit")
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_BYTES:
                            raise FetchFailure("document exceeds the 20 MB limit")
                        if time.monotonic() > deadline:
                            raise FetchFailure("download took too long")
                    kind = classify(response.headers.get("content-type", ""), bytes(data))
                    return Fetched(url, kind, bytes(data), response.charset_encoding)
            except httpx.HTTPError:
                raise FetchFailure("connection or timeout error") from None
        raise FetchFailure("too many redirects")
