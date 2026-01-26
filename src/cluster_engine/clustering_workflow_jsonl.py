import os
from langgraph.graph import StateGraph, END

# Import node functions
from .nodes.load_data import load_data
from .nodes.embed_prompts import embed_prompts
from .nodes.optimize_kmeans import optimize_kmeans
from .nodes.cluster_data import cluster_data
from .nodes.hierarchical_clustering import hierarchical_clustering
from .nodes.label_clusters import label_clusters
from .nodes.save_results import save_results

# Import the state type from the shared types module
from .nodes.types import ClusteringState


# Build the LangGraph workflow
def build_clustering_workflow():
    """Build and compile the clustering workflow graph."""

    # Create the graph
    workflow = StateGraph(ClusteringState)

    # Add nodes
    workflow.add_node("load_data", load_data)
    workflow.add_node("embed_prompts", embed_prompts)
    workflow.add_node("optimize_kmeans", optimize_kmeans)
    workflow.add_node("cluster_data", cluster_data)
    workflow.add_node("hierarchical_clustering", hierarchical_clustering)
    workflow.add_node("label_clusters", label_clusters)
    workflow.add_node("save_results", save_results)

    # Define the edges (workflow sequence)
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "embed_prompts")
    workflow.add_edge("embed_prompts", "optimize_kmeans")
    workflow.add_edge("optimize_kmeans", "cluster_data")
    workflow.add_edge("cluster_data", "hierarchical_clustering")
    workflow.add_edge("hierarchical_clustering", "label_clusters")
    workflow.add_edge("label_clusters", "save_results")
    workflow.add_edge("save_results", END)

    # Compile the graph
    app = workflow.compile()

    return app


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("LangGraph-based Clustering Workflow")
    print("=" * 80)

    # Build and run the workflow
    app = build_clustering_workflow()

    # Initialize state with configuration
    # Set use_llm_labeling=True to enable LLM-based cluster labeling
    initial_state = {
        'use_llm_labeling': False  # Set to True to enable LLM labeling
    }

    # Execute the workflow
    final_state = app.invoke(initial_state)

    print("=" * 80)
    print("Workflow completed successfully!")
    print(f"Processed {len(final_state['prompts'])} prompts")
    print(f"Created {len(final_state['final_clusters'])} final clusters")
    print("=" * 80)
