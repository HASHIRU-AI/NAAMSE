from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os
import json
import random
import subprocess
import platform
from itertools import islice

from .data_access import DataSource, create_data_source

def _count_lines_cross_platform(file_path: str) -> int:
    """
    Count lines in a file using cross-platform command-line tools.
    
    Args:
        file_path: Path to the file to count lines in
        
    Returns:
        Number of lines in the file
    """
    system = platform.system().lower()
    
    if system == 'windows':
        # Use Windows find command: find /c /v "" file
        # This counts all lines including empty ones
        result = subprocess.run(['find', '/c', '/v', '""', file_path],
                              capture_output=True, text=True, check=True)
        # Output format: "---------- FILE: 123"
        line_count = int(result.stdout.strip().split()[-1])
    else:
        # Use Unix wc command
        result = subprocess.run(['wc', '-l', file_path],
                              capture_output=True, text=True, check=True)
        line_count = int(result.stdout.split()[0])
    
    return line_count
_MODEL_CACHE = {}


def _get_cached_model(device: str = None) -> SentenceTransformer:
    """
    Get a cached SentenceTransformer model to avoid reloading on every call.
    
    Args:
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
    
    Returns:
        Cached SentenceTransformer instance
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    cache_key = f'all-MiniLM-L6-v2_{device}'
    
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    return _MODEL_CACHE[cache_key]


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

    # Encode query prompt using cached model
    model = _get_cached_model(device)
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

    # Check for duplicates before proceeding
    existing_prompts, _ = data_source.get_prompts_and_sources()
    if new_prompt in existing_prompts:
        print(f"⚠️  Skipping duplicate prompt: {new_prompt[:50]}...")
        # Return a result indicating the prompt was skipped
        return {
            'prompt': new_prompt,
            'source': source,
            'cluster_id': None,
            'cluster_label': None,
            'centroid_coord': None,
            'distance_to_centroid': None,
            'embedding_index': None
        }

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

    # Embed the new prompt using cached model
    print(f"Embedding new prompt on {device}...")
    model = _get_cached_model(device)
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

    # Get cluster label from existing prompts in the assigned cluster
    cluster_path = nearest_cluster_id
    cluster_label = f"Cluster_{nearest_cluster_id.replace('/', '_')}"

    try:
        cluster_prompts = data_source.get_prompts_by_cluster(nearest_cluster_id)
        if cluster_prompts:
            cluster_label = cluster_prompts[0].get('cluster_label', cluster_label)
    except Exception as e:
        print(f"Warning: Could not retrieve cluster label from data source: {e}")

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
        'cluster_id': cluster_path,
        'cluster_label': cluster_label,
        'centroid_coord': nearest_centroid.tolist(),
        'distance_to_centroid': float(nearest_distance),
        'embedding_index': len(existing_embeddings)
    }

    print("✅ Prompt added successfully!")
    return result


def get_random_prompt(data_source: Optional[DataSource] = None, _cached_line_count: dict = {}) -> Dict[str, Any]:
    """
    Get a random prompt from the corpus using fast random access.
    
    This function uses subprocess `wc -l` to count lines once and caches the count,
    then uses random access to fetch a single line without loading embeddings.

    Args:
        data_source: Data source to use (if None, uses JSONL data source)
        _cached_line_count: Internal cache for line count (do not set manually)

    Returns:
        Dictionary containing a random prompt with its source and index
    """
    if data_source is None:
        data_source = create_data_source('jsonl')

    # Assuming data_source is JSONLDataSource for direct file access
    corpus_file = data_source.corpus_file

    if not os.path.exists(corpus_file):
        raise ValueError("No prompts found in the corpus")

    # Cache line count using cross-platform line counting
    if corpus_file not in _cached_line_count:
        line_count = _count_lines_cross_platform(corpus_file)
        # line_count = 100000 # Placeholder for cross-platform line count
        _cached_line_count[corpus_file] = line_count
    else:
        line_count = _cached_line_count[corpus_file]
    
    if line_count == 0:
        raise ValueError("No prompts found in the corpus")
    
    # Pick a random line index
    target_index = random.randint(0, line_count - 1)
    
    # Read only that specific line using islice for efficiency
    with open(corpus_file, 'r') as f:
        line = next(islice(f, target_index, None))
        selected_data = json.loads(line.strip())
    
    # Build result
    result = {
        'prompt': selected_data['messages'][0]['content'],
        'source': selected_data['source'],
        'index': target_index
    }

    # Add cluster info if available
    if 'cluster_id' in selected_data:
        result['cluster_id'] = selected_data['cluster_id']
    if 'cluster_label' in selected_data:
        result['cluster_label'] = selected_data['cluster_label']
    if 'centroid_coord' in selected_data:
        result['centroid_coord'] = selected_data['centroid_coord']

    return result

def get_cluster_id_for_prompt(prompt: List[str], data_source: Optional[DataSource] = None,
                               device: str = None) -> Optional[str]:
    """
    Find the nearest prompt to the given prompt, get its cluster_id, and return that cluster_id.

    Args:
        prompt: The input prompt to find the nearest match for
        data_source: Data source to use (if None, creates JSONL data source)
        device: Device to use for encoding

    Returns:
        Cluster id of the nearest prompt, or None if not found
    """
    # Find the nearest prompt
    prompt_text = prompt[0] if len(prompt) > 0 else ""
    nearest = find_nearest_prompts(prompt_text, n=1, data_source=data_source, device=device)
    if not nearest:
        return None
    
    nearest_data = nearest[0]
    cluster_id = nearest_data.get('cluster_id')
    if not cluster_id:
        return None
    
    return cluster_id

def get_human_readable_cluster_info(cluster_id: str, lookup_file: str = 'cluster_lookup_table.json') -> Optional[Dict[str, str]]:
    """
    Given a cluster_id, look up its human-readable label and description from a lookup table.

    Args:
        cluster_id: The cluster_id to look up
        lookup_file: Path to the cluster lookup table JSON file
        
    Returns:
        Updated cluster information dictionary with 'label' and 'description', or None if not found
    """
    print(f"Getting description for cluster_id: {cluster_id}")
    # Load lookup table
    lookup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cluster_lookup_table.json')
    
    if not os.path.exists(lookup_file):
        return None
    
    with open(lookup_file, 'r') as f:
        lookup_table = json.load(f)
    
    # Find the most specific match
    cluster_info = None
    max_parts = 0
    for key, val in lookup_table.items():
        if cluster_id.startswith(key):
            parts = len(key.split('/'))
            if parts > max_parts:
                max_parts = parts
                if isinstance(val, dict):
                    cluster_info = val
                else:
                    cluster_info = {'label': val, 'description': ''}
    
    # If no hierarchical match found, try centroid-based nearest neighbor
    if cluster_info is None:
        centroids_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'centroids.pkl')
        if os.path.exists(centroids_file):
            with open(centroids_file, 'rb') as f:
                centroids = pickle.load(f)
            if cluster_id in centroids:
                target_centroid = np.array(centroids[cluster_id])
                min_dist = float('inf')
                closest_info = None
                for key, val in lookup_table.items():
                    if key in centroids:
                        coord = np.array(centroids[key])
                        dist = np.linalg.norm(target_centroid - coord)
                        if dist < min_dist:
                            min_dist = dist
                            closest_info = val
                if closest_info:
                    cluster_info = closest_info
    
    return cluster_info