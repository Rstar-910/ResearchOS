import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from researchos.models import ResearchState, SearchEvent
from researchos.planning import Planner
from researchos.search import SearchFailure, ToolRegistry, merge_sources
from researchos.storage import Repository


class Discovery:
    """Single-process worker with persisted checkpoints and bounded admission."""

    def __init__(
        self, repository: Repository, registry: ToolRegistry, planner: Planner | None = None
    ):
        self.repository, self.registry, self.planner = repository, registry, planner
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discovery")
        self.lock = Lock()
        self.active = False

    def start(self, state: ResearchState) -> ResearchState:
        with self.lock:
            if self.active:
                raise ValueError("Discovery is busy. Wait for the current session to finish.")
            if not self.registry.providers:
                raise ValueError("Discovery is disabled; enable DISCOVERY_ENABLED.")
            if len(state.search_history) >= state.budget.max_searches:
                raise ValueError("The session search budget is exhausted.")
            if len(state.sources) >= state.budget.max_sources:
                raise ValueError("The session source budget is exhausted.")
            if state.status in {"reading", "analyzing"}:
                raise ValueError("This session is busy reading or analyzing sources.")
            if state.status not in {"awaiting_providers", "discovery_failed", "interrupted"}:
                raise ValueError("Discovery is finished. Reading the sources is next.")
            if state.status == "searching":
                raise ValueError("This session is already searching.")
            state.status = "searching"
            state.record(
                "searching", "Source discovery started. Search results are not verified evidence."
            )
            if not self.repository.claim_search(state):
                raise ValueError("Session unavailable or already searching.")
            self.active = True
            self.executor.submit(self.run, state.model_copy(deep=True))
            return state

    def run(self, state: ResearchState) -> None:
        try:
            completed = {
                (event.provider, event.query)
                for event in state.search_history
                if event.status == "completed"
            }
            suspended = set()
            if self.planner and state.plan_source == "template" and not state.search_history:
                state.record("planning", f"Asking {self.planner.llm.model} for a research plan.")
                if not self.repository.save(state):
                    return
                self.planner.refine(state)
                if not self.repository.save(state):
                    return
            web = any(p.source_type == "web" for p in self.registry.providers)
            if not web:
                state.record(
                    "provider_unavailable", "Web search skipped: SEARXNG_URL is not configured."
                )
            # Topic qualifiers give template tasks a deterministic query; model-planned
            # tasks carry their own.
            qualifiers = [
                "",
                "methods",
                "benchmark evaluation",
                "implementation",
                "limitations criticism",
                "recent research",
            ]
            stopwords = {
                "what",
                "are",
                "the",
                "is",
                "how",
                "does",
                "do",
                "can",
                "of",
                "to",
                "in",
                "a",
                "an",
                "and",
                "for",
                "with",
                "as",
                "whether",
            }
            terms = [
                word
                for word in re.findall(r"[\w-]+", state.question)
                if word.lower() not in stopwords
            ]
            base_query = " ".join(terms[:24]) or state.question[:200]
            for index, task in enumerate(state.tasks):
                if (
                    len(state.search_history) >= state.budget.max_searches
                    or len(state.sources) >= state.budget.max_sources
                ):
                    # Tasks the budget never reached stay pending rather than failed.
                    break
                task.status = "running"
                if not self.repository.save(state):
                    return
                query = task.query or f"{base_query} {qualifiers[index % len(qualifiers)]}".strip()
                task_found_sources = False
                for provider in self.registry.for_types(task.source_types):
                    if provider.name in suspended:
                        continue
                    if (provider.name, query) in completed:
                        task_found_sources = True
                        continue
                    for attempt in range(2):
                        if (
                            len(state.search_history) >= state.budget.max_searches
                            or len(state.sources) >= state.budget.max_sources
                        ):
                            break
                        event = SearchEvent(provider=provider.name, query=query, task_id=task.id)
                        state.search_history.append(event)
                        state.record("search_request", f"Searching {provider.name}: {query}")
                        if not self.repository.save(state):
                            return
                        try:
                            found = provider.search(query, min(5, state.budget.max_sources))
                            merge_sources(state.sources, found, state.budget.max_sources)
                            event.status = "completed"
                            event.results_count = len(found)
                            completed.add((provider.name, query))
                            task_found_sources = True
                            state.record(
                                "search_results",
                                f"{provider.name}: {len(found)} results; "
                                f"{len(state.sources)} unique sources saved.",
                            )
                            if not self.repository.save(state):
                                return
                            break
                        except SearchFailure as error:
                            event.status = "failed"
                            event.error = str(error)
                            state.errors.append(str(error))
                            state.record("search_error", str(error))
                            if not self.repository.save(state):
                                return
                            if not error.retryable or attempt == 1:
                                if error.suspend_provider:
                                    suspended.add(provider.name)
                                    state.record(
                                        "provider_paused",
                                        f"{provider.name} skipped for the rest of this run "
                                        "after rejection.",
                                    )
                                    if not self.repository.save(state):
                                        return
                                break
                            if len(state.search_history) < state.budget.max_searches:
                                time.sleep(error.retry_after)
                task.status = "completed" if task_found_sources else "failed"
                if not self.repository.save(state):
                    return
            state.status = "discovery_complete" if state.sources else "discovery_failed"
            reason = "Search plan finished."
            if len(state.sources) >= state.budget.max_sources:
                reason = "Source limit reached."
            elif len(state.search_history) >= state.budget.max_searches:
                reason = "Search attempt limit reached."
            state.record(
                state.status,
                f"{reason} {len(state.sources)} unique sources saved. "
                "Documents have not been read yet.",
            )
            self.repository.save(state)
        except Exception:
            state.status = "discovery_failed"
            for event in state.search_history:
                if event.status == "running":
                    event.status = "failed"
                    event.error = "Unexpected provider failure."
            state.record(
                "discovery_failed", "Unexpected discovery failure. Saved sources are retained."
            )
            self.repository.save(state)
        finally:
            with self.lock:
                self.active = False

    def close(self) -> None:
        self.executor.shutdown(wait=True)
