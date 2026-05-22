from langgraph.graph import StateGraph, END

from state import ResearchState
from research_agents import (
    data_agent,
    sentiment_agent,
    report_agent
)

graph = StateGraph(ResearchState)

graph.add_node("data_agent", data_agent)
graph.add_node("sentiment_agent", sentiment_agent)
graph.add_node("report_agent", report_agent)

graph.set_entry_point("data_agent")

graph.add_edge("data_agent", "sentiment_agent")
graph.add_edge("sentiment_agent", "report_agent")
graph.add_edge("report_agent", END)

app = graph.compile()