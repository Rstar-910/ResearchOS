from typing import Literal

from pydantic import BaseModel, Field, field_validator

from researchos.llm import LLM
from researchos.models import ResearchState, Task, now

MAX_KEYWORDS = 6

SYSTEM = """You plan literature and web research. Given a research question, break it into \
5 to 8 subquestions that together answer it. Cover definitions and scope, primary methods, \
benchmarks or evidence, practical implementations, criticisms or conflicting findings, and \
recent developments, where each applies to this question.

For each subquestion give:
- question: the subquestion, specific to this topic.
- query: 3 to 6 search keywords for it (never more than 6), with no punctuation, quotes, \
or boolean operators. Academic search requires every keyword to match, so use only the few \
terms that papers on this would certainly contain, and leave out years.
- source_types: "papers" for academic search, "web" for general web search, or both.

Reply with only a JSON object of the form {"tasks": [{"question": ..., "query": ..., \
"source_types": [...]}]}."""


class PlannedTask(BaseModel):
    question: str = Field(min_length=5, max_length=400)
    query: str = Field(min_length=2, max_length=120)
    source_types: list[Literal["papers", "web"]] = Field(min_length=1, max_length=2)

    @field_validator("query")
    @classmethod
    def few_keywords(cls, value: str) -> str:
        # Academic search requires every keyword to match, so long queries find nothing.
        # Trimming keeps an otherwise good plan instead of paying for a second attempt.
        return " ".join(value.split()[:MAX_KEYWORDS])


class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(min_length=3, max_length=8)


class Planner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def refine(self, state: ResearchState) -> None:
        """Replace the template plan with a model-written one; keep the template on failure."""
        remaining = state.budget.max_tokens - state.token_usage
        user = f"Today's date: {now():%Y-%m-%d}\n\nResearch question: {state.question}"
        plan, used, error = self.llm.structured(SYSTEM, user, Plan, remaining)
        state.token_usage += used
        if plan is None:
            state.record(
                "planning_fallback", f"Model planning failed ({error}); using the template plan."
            )
            return
        state.tasks = [
            Task(
                question=" ".join(task.question.split()),
                query=" ".join(task.query.split()),
                source_types=list(dict.fromkeys(task.source_types)),
            )
            for task in plan.tasks
        ]
        state.plan_source = "model"
        state.record(
            "plan_ready",
            f"{self.llm.model} wrote a {len(state.tasks)}-task plan ({used:,} tokens).",
        )
