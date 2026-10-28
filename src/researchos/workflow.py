from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from researchos.models import ResearchRequest, ResearchState, Task
from researchos.storage import Repository


class GraphState(TypedDict):
    session: ResearchState


def plan(state: GraphState) -> GraphState:
    session = state["session"].model_copy(deep=True)
    session.status = "planning"
    session.record("planning", "Building a template research plan; no LLM has been called.")
    topics = [
        ("Definitions and scope", ["web", "papers"]),
        ("Primary methods and original research", ["papers"]),
        ("Benchmarks, datasets, and evaluation", ["papers", "web"]),
        ("Implementations and practical constraints", ["web"]),
        ("Conflicting findings and methodological limitations", ["papers", "web"]),
        ("Recent developments and unresolved questions", ["papers", "web"]),
    ]
    session.tasks = [
        Task(question=f"{topic}: {session.question}", source_types=types) for topic, types in topics
    ]
    session.status = "awaiting_providers"
    session.record(
        "awaiting_providers",
        "Plan saved and ready for source discovery. No model provider is connected; "
        "no findings have been generated.",
    )
    return {"session": session}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("plan", plan)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", END)
    return graph.compile()


class ResearchManager:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.graph = build_graph()

    def create(self, request: ResearchRequest) -> ResearchState:
        session = ResearchState(question=request.question, budget=request.budget)
        session.record("created", "Research session created.")
        result = self.graph.invoke({"session": session})["session"]
        self.repository.create(result)
        return result
