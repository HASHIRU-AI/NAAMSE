import json
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import pickle
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 5: Hierarchical clustering
def hierarchical_clustering(state: ClusteringState) -> ClusteringState:
    """Recursively cluster data to create hierarchical structure."""

    def get_hierarchical_clusters(clusters, best_params, depth=0, parent_path=""):
        result = []
        for idx, cluster in enumerate(clusters):
            cluster_size = len(cluster['prompts'])
            current_path = f"{parent_path}/cluster_{idx}" if parent_path else f"cluster_{idx}"

            if cluster_size <= 5:
                print(f"{'  ' * depth}Cluster size {cluster_size} <= 5, stopping recursion")
                result.append({
                    'path': current_path,
                    'prompts': cluster['prompts'],
                    'sources': cluster['sources'],
                    'centroid': cluster['centroid']
                })
                continue

            sub_embeddings = np.array(cluster['embeddings'])
            # Use smaller number of clusters for recursion
            n_sub_clusters = max(2, min(5, cluster_size // 10))

            print(f"{'  ' * depth}Cluster size {cluster_size}: attempting {n_sub_clusters} subclusters at {current_path}")

            kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(sub_embeddings)

            # Check if K-means created meaningful subclusters
            unique_labels = set(sub_labels)
            if len(unique_labels) > 1:
                sub_clusters = []
                for sl in unique_labels:
                    indices = np.where(sub_labels == sl)[0]
                    if len(indices) > 0:  # Only add non-empty clusters
                        sub_clusters.append({
                            'embeddings': sub_embeddings[indices].tolist(),
                            'prompts': [cluster['prompts'][i] for i in indices],
                            'sources': [cluster['sources'][i] for i in indices],
                            'centroid': kmeans.cluster_centers_[sl].tolist()
                        })
                # Recursively cluster
                sub_result = get_hierarchical_clusters(sub_clusters, best_params, depth + 1, current_path)
                result.extend(sub_result)
            else:
                print(f"{'  ' * depth}Cannot split further, keeping as final cluster")
                result.append({
                    'path': current_path,
                    'prompts': cluster['prompts'],
                    'sources': cluster['sources'],
                    'centroid': cluster['centroid']
                })
        return result

    clusters_file = f"{state['checkpoint_dir']}/final_clusters.pkl"

    if os.path.exists(clusters_file):
        print("Loading cached final clusters...")
        with open(clusters_file, 'rb') as f:
            final_clusters = pickle.load(f)
        print(f"Loaded {len(final_clusters)} final subclusters from cache")
    else:
        final_clusters = get_hierarchical_clusters(state['initial_clusters'], state['best_params'])
        print(f"Final subclusters: {len(final_clusters)}")

        # Save clusters for future runs
        with open(clusters_file, 'wb') as f:
            pickle.dump(final_clusters, f)
        print(f"Saved clusters to {clusters_file}")

    print(f"Total final clusters: {len(final_clusters)}")

    return {
        **state,
        'final_clusters': final_clusters
    }