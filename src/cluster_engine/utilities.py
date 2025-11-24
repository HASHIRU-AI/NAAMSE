from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os
import json
import random

from .data_access import DataSource, create_data_source


def find_nearest_prompts(query_prompt: str, n: int = 1, data_source: Optional[DataSource] = None,
                        device: str = None) -> List[Dict[str, Any]]:
    """
    Find the n nearest prompts to a given query prompt.

    Args:
        query_prompt: The prompt to find similar prompts for
        n: Number of nearest prompts to return
        data_source: Data source to use (if None, creates JSONL data source)
        device: Device to use for encoding ('cpu', 'cuda', 'mps', or None for auto-detect)

    Returns:
        List of dictionaries containing the nearest prompts with their similarity scores
        and cluster information if available in the corpus file
    """
    if data_source is None:
        data_source = create_data_source('jsonl')

    # Load embeddings
    embeddings = data_source.get_embeddings()

    # Determine device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Encode query prompt
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    query_embedding = model.encode([query_prompt])[0]

    # Calculate cosine similarities
    # Normalize vectors for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    # Get top n indices
    top_n_indices = np.argsort(similarities)[::-1][:n]

    # Now load only the required prompts and cluster info
    corpus_file = data_source.corpus_file
    top_indices_set = set(top_n_indices)
    selected_data = {}

    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num in top_indices_set:
                data = json.loads(line.strip())
                selected_data[line_num] = data

    # Build results
    results = []
    for idx in top_n_indices:
        data = selected_data[idx]
        result = {
            'prompt': data['messages'][0]['content'],
            'source': data['source'],
            'similarity': float(similarities[idx]),
            'index': int(idx)
        }
        # Add cluster info if available
        if 'cluster_id' in data:
            result['cluster_id'] = data['cluster_id']
        if 'cluster_label' in data:
            result['cluster_label'] = data['cluster_label']
        if 'centroid_coord' in data:
            result['centroid_coord'] = data['centroid_coord']
        results.append(result)

    return results


def get_prompts_by_cluster(cluster_id: str, data_source: Optional[DataSource] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all prompts belonging to a specific cluster.

    Args:
        cluster_id: The cluster_id to filter by (e.g., 'cluster_0', 'cluster_0/cluster_1')
        data_source: Data source to use (if None, creates JSONL data source)

    Returns:
        List of dictionaries containing prompts and their metadata from the specified cluster
    """
    if data_source is None:
        data_source = create_data_source('jsonl')

    cluster_prompts = data_source.get_prompts_by_cluster(cluster_id)
    print(f"Found {len(cluster_prompts)} prompts in cluster '{cluster_id}'")
    return cluster_prompts


def add_prompt_to_clusters(new_prompt: str, source: str = 'NAAMSE_mutation',
                          data_source: Optional[DataSource] = None,
                          centroids_file: str = 'centroids.pkl',
                          device: str = None) -> Dict[str, Any]:
    """
    Add a new prompt to the existing cluster structure without re-clustering.

    Approach:
    1. Embed the new prompt using the same model
    2. Find the nearest cluster centroid
    3. Update the data source with cluster metadata
    4. Update embeddings

    Args:
        new_prompt: The new prompt to add
        source: Source identifier for the prompt
        data_source: Data source to use (if None, creates JSONL data source)
        centroids_file: Path to centroids pickle file (relative to script directory if not absolute)
        device: Device to use for encoding

    Returns:
        Dictionary with information about where the prompt was added
    """
    if data_source is None:
        data_source = create_data_source('jsonl')

    # Make centroids_file path relative to script directory if not absolute
    if not os.path.isabs(centroids_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        centroids_file = os.path.join(script_dir, centroids_file)

    # Load centroids
    if not os.path.exists(centroids_file):
        raise FileNotFoundError(f"Centroids file not found: {centroids_file}. Run clustering first.")

    with open(centroids_file, 'rb') as f:
        centroids = pickle.load(f)

    # Determine device
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Embed the new prompt
    print(f"Embedding new prompt on {device}...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    new_embedding = model.encode([new_prompt])[0]

    # Find nearest centroid
    distances = {}
    for cluster_id, centroid in centroids.items():
        dist = np.linalg.norm(new_embedding - centroid)
        distances[cluster_id] = dist

    nearest_cluster_id = min(distances, key=distances.get)
    nearest_distance = distances[nearest_cluster_id]
    nearest_centroid = centroids[nearest_cluster_id]

    print(f"Nearest cluster: {nearest_cluster_id} (distance: {nearest_distance:.4f})")

    # Load cluster labels and paths to get cluster metadata
    cluster_labels = {}
    cluster_paths = {}

    # Try to load from checkpoints directory (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_file = os.path.join(script_dir, 'checkpoints', 'cluster_labels.json')
    clusters_file = os.path.join(script_dir, 'checkpoints', 'final_clusters.pkl')

    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            cluster_labels = {int(k): v for k, v in json.load(f).items()}

    if os.path.exists(clusters_file):
        with open(clusters_file, 'rb') as f:
            final_clusters = pickle.load(f)
        if nearest_cluster_id < len(final_clusters):
            cluster_paths = {i: fc['path'] for i, fc in enumerate(final_clusters)}

    cluster_label = cluster_labels.get(nearest_cluster_id, f"Cluster_{nearest_cluster_id}")
    cluster_path = cluster_paths.get(nearest_cluster_id, f"cluster_{nearest_cluster_id}")

    # Create cluster info for the new prompt
    cluster_info = {
        "cluster_id": cluster_path,
        "cluster_label": cluster_label,
        "centroid_coord": nearest_centroid.tolist()
    }

    # Add to data source
    data_source.add_prompt(new_prompt, source, cluster_info)

    # Update embeddings
    existing_embeddings = data_source.get_embeddings()
    updated_embeddings = np.vstack([existing_embeddings, new_embedding])
    data_source.save_embeddings(updated_embeddings)

    result = {
        'prompt': new_prompt,
        'source': source,
        'assigned_cluster_id': cluster_path,
        'assigned_cluster_label': cluster_label,
        'distance_to_centroid': float(nearest_distance),
        'embedding_index': len(existing_embeddings)
    }

    print("✅ Prompt added successfully!")
    return result


def get_random_prompt(data_source: Optional[DataSource] = None) -> Dict[str, Any]:
    """
    Get a random prompt from the corpus.

    Args:
        data_source: Data source to use (if None, uses JSONL data source)

    Returns:
        Dictionary containing a random prompt with its source and index
    """
    if data_source is None:
        data_source = create_data_source('jsonl')

    # Assuming data_source is JSONLDataSource for direct file access
    corpus_file = data_source.corpus_file

    if not os.path.exists(corpus_file):
        raise ValueError("No prompts found in the corpus")

    selected_data = None
    selected_index = 0
    line_count = 0

    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line.strip())
            # Reservoir sampling: keep this item with probability 1/(line_num+1)
            if random.random() < 1.0 / (line_num + 1):
                selected_data = data
                selected_index = line_num
            line_count = line_num + 1

    if selected_data is None:
        raise ValueError("No prompts found in the corpus")

    # Build result
    result = {
        'prompt': selected_data['messages'][0]['content'],
        'source': selected_data['source'],
        'index': selected_index
    }

    # Add cluster info if available
    if 'cluster_id' in selected_data:
        result['cluster_id'] = selected_data['cluster_id']
    if 'cluster_label' in selected_data:
        result['cluster_label'] = selected_data['cluster_label']
    if 'centroid_coord' in selected_data:
        result['centroid_coord'] = selected_data['centroid_coord']

    return result