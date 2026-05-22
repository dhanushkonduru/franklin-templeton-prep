from langgraph.graph import StateGraph, END

from langgraph.checkpoint.memory import MemorySaver

from state.research_state import ResearchState
from agents.retrieval_agent import (
    retrieval_agent
)

from agents.data_gatherer import data_gatherer
from agents.fundamental_analyst import fundamental_analyst
from agents.sentiment_analyst import sentiment_analyst
from agents.risk_analyst import risk_analyst
from agents.report_writer import report_writer

from agents.critic_agent import critic_agent

from orchestration.routing import review_router

graph = StateGraph(ResearchState)

graph.add_node(
    "data_gatherer",
    data_gatherer
)

graph.add_node(
    "fundamental_analyst",
    fundamental_analyst
)
graph.add_node(
    "retrieval_agent",
    retrieval_agent
)

graph.add_node(
    "sentiment_analyst",
    sentiment_analyst
)

graph.add_node(
    "risk_analyst",
    risk_analyst
)

graph.add_node(
    "report_writer",
    report_writer
)

graph.add_node(
    "critic_agent",
    critic_agent
)

graph.set_entry_point(
    "data_gatherer"
)

graph.add_edge(
    "data_gatherer",
    "retrieval_agent"
)

graph.add_edge(
    "retrieval_agent",
    "fundamental_analyst"
)

graph.add_edge(
    "fundamental_analyst",
    "sentiment_analyst"
)

graph.add_edge(
    "sentiment_analyst",
    "risk_analyst"
)

graph.add_edge(
    "risk_analyst",
    "report_writer"
)

graph.add_edge(
    "report_writer",
    "critic_agent"
)

graph.add_conditional_edges(
    "critic_agent",
    review_router,
    {
        "report_writer": "report_writer",
        "end": END
    }
)

checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer
)