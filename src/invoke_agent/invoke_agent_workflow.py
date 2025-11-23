from src.invoke_agent.invoke_agent_state import InvokeAgentWorkflowState
from src.invoke_agent.nodes.invoke_agent import invoke_a2a_agent
from langgraph.graph import StateGraph, START, END


# --- Build the Main Mutation Engine Graph ---
main_graph_builder = StateGraph(InvokeAgentWorkflowState)

main_graph_builder.add_node("InvokeA2AAgent", invoke_a2a_agent)
# Define Edges
main_graph_builder.add_edge(START, "InvokeA2AAgent")
main_graph_builder.add_edge("InvokeA2AAgent", END)
# --- Compile the graph for export ---
invoke_agent_graph = main_graph_builder.compile()
