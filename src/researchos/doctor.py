"""One-request-per-provider diagnostic. Does not print secrets or change sessions."""

import argparse

import httpx
from pydantic import ValidationError

from researchos.config import Settings
from researchos.llm import LLMFailure, configured_llm
from researchos.search import SearchFailure, configured_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="electron", help="Use a short keyword query")
    parser.add_argument(
        "--provider",
        choices=["all", "arxiv", "semantic_scholar", "searxng", "llm"],
        default="all",
    )
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValidationError as error:
        print(
            "Invalid configuration fields: "
            + ", ".join(str(item["loc"][0]) for item in error.errors())
        )
        return 2
    print(
        "Semantic Scholar key: " + ("configured" if settings.semantic_scholar_api_key else "absent")
    )
    print("SearXNG URL: " + ("configured" if settings.searxng_url else "absent"))
    if not settings.discovery_enabled:
        print("DISCOVERY_ENABLED=false; diagnostics will still probe selected providers.")
    settings = settings.model_copy(update={"discovery_enabled": True})
    failed = False
    with httpx.Client(
        timeout=settings.search_timeout, headers={"User-Agent": "ResearchOS/0.1"}
    ) as client:
        providers = configured_registry(settings, client).providers
        providers = [p for p in providers if args.provider in {"all", p.name}]
        llm = configured_llm(settings, client) if args.provider in {"all", "llm"} else None
        if llm:
            try:
                _, used = llm.chat([{"role": "user", "content": "Reply with: ok"}], None, 512)
                print(f"llm: OK ({llm.model}, {used} tokens)")
            except LLMFailure as error:
                failed = True
                print(f"llm: {error}")
        elif args.provider in {"all", "llm"}:
            print("llm: not configured. Set OLLAMA_API_KEY (cloud) or OLLAMA_BASE_URL (local).")
        if not providers and args.provider == "llm":
            return int(failed or llm is None)
        if not providers:
            print("Selected provider is not configured. Set SEARXNG_URL to test web search.")
            return 2
        for provider in providers:
            try:
                results = provider.search(args.query, 1)
                print(f"{provider.name}: OK ({len(results)} results)")
            except SearchFailure as error:
                failed = True
                print(str(error))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
