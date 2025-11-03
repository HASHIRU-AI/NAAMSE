from typing import TypedDict, List, Dict, Any, Optional
import numpy as np


# Define the state that flows through the graph
class ClusteringState(TypedDict):
    # Input data
    prompts: List[str]
    sources: List[str]

    # Embeddings
    embeddings: Optional[np.ndarray]
    embeddings_file: str

    # K-means parameters
    best_params: Optional[Dict[str, Any]]

    # Clustering results
    labels: Optional[np.ndarray]
    initial_clusters: Optional[List[Dict[str, Any]]]
    final_clusters: Optional[List[Dict[str, Any]]]

    # Cluster labels
    cluster_labels: Optional[Dict[int, str]]

    # Configuration
    checkpoint_dir: str
    device: str
    use_llm_labeling: bool