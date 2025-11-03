import json
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 4: Perform initial clustering
def cluster_data(state: ClusteringState) -> ClusteringState:
    """Perform initial K-means clustering on full corpus."""
    print("Clustering full corpus with best params...")

    kmeans = KMeans(n_clusters=state['best_params']['n_clusters'], random_state=42, n_init=10)
    labels = kmeans.fit_predict(state['embeddings'])

    # Get centroids from K-means
    centroids = {i: kmeans.cluster_centers_[i] for i in range(kmeans.n_clusters)}

    # Create initial clusters
    initial_clusters = []
    for label in range(kmeans.n_clusters):
        indices = np.where(labels == label)[0]
        if len(indices) > 0:  # Only add non-empty clusters
            initial_clusters.append({
                'embeddings': state['embeddings'][indices].tolist(),
                'prompts': [state['prompts'][i] for i in indices],
                'sources': [state['sources'][i] for i in indices],
                'centroid': centroids[label].tolist()
            })

    print(f"Created {len(initial_clusters)} initial clusters")

    return {
        **state,
        'labels': labels,
        'initial_clusters': initial_clusters
    }