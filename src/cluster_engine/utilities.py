from typing import List, Dict, Any, Optional

from src.cluster_engine.data_access.data_source import DataSource


def find_nearest_prompts(query_prompt: str, n: int = 1, data_source: Optional[DataSource] = None,
                         device: str = None, seed=None) -> List[Dict[str, Any]]:
    """
    Find the n nearest prompts to a given query prompt.

    Args:
        query_prompt: The prompt to find similar prompts for
        n: Number of nearest prompts to return
        data_source: Data source to use (if None, creates SQLite data source)
        device: Device to use for encoding ('cpu', 'cuda', 'mps', or None for auto-detect)
        seed: Random seed for reproducible sampling

    Returns:
        List of dictionaries containing the nearest prompts with their similarity scores
        and cluster information if available in the corpus file
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    return data_source.find_nearest_prompts(query_prompt, n, device, seed)


def get_prompts_by_cluster(cluster_id: str, data_source: Optional[DataSource] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all prompts belonging to a specific cluster.

    Args:
        cluster_id: The cluster_id to filter by (e.g., 'cluster_0', 'cluster_0/cluster_1')
        data_source: Data source to use (if None, creates SQLite data source)

    Returns:
        List of dictionaries containing prompts and their metadata from the specified cluster
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    cluster_prompts = data_source.get_prompts_by_cluster(cluster_id)
    print(f"Found {len(cluster_prompts)} prompts in cluster '{cluster_id}'")
    return cluster_prompts


def add_prompt_to_clusters(new_prompt: str,
                           data_source: Optional[DataSource] = None,
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
        data_source: Data source to use (if None, creates SQLite data source)
        device: Device to use for encoding

    Returns:
        Dictionary with information about where the prompt was added
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    return data_source.add_prompt_to_clusters(new_prompt, device)


def get_random_prompt(data_source: Optional[DataSource] = None, seed=None) -> Dict[str, Any]:
    """
    Get a random prompt from the corpus using fast random access.

    This function uses reservoir sampling to fetch a single line without loading embeddings.

    Args:
        data_source: Data source to use (if None, uses SQLite data source)
        seed: Random seed for reproducible sampling

    Returns:
        Dictionary containing a random prompt with its source and index
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    return data_source.get_random_prompt(seed=seed)


def get_cluster_id_for_prompt(prompt: List[str], data_source: Optional[DataSource] = None,
                              device: str = None) -> Optional[str]:
    """
    Find the nearest prompt to the given prompt, get its cluster_id, and return that cluster_id.

    Args:
        prompt: The input prompt to find the nearest match for
        data_source: Data source to use (if None, creates SQLite data source)
        device: Device to use for encoding

    Returns:
        Cluster id of the nearest prompt, or None if not found
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    return data_source.get_cluster_id_for_prompt(prompt, device)


def get_human_readable_cluster_info(cluster_id: str,
                                    data_source: Optional[DataSource] = None) -> Optional[Dict[str, str]]:
    """
    Given a cluster_id, look up its human-readable label and description from a lookup table.

    Args:
        cluster_id: The cluster_id to look up
        data_source: Data source to use (if None, creates SQLite data source)

    Returns:
        Updated cluster information dictionary with 'label' and 'description', or None if not found
    """
    if data_source is None:
        raise NotImplementedError(
            "Data source must be provided for this operation.")
        # data_source = create_data_source('sqlite')

    return data_source.get_human_readable_cluster_info(cluster_id)

# Factory function to create data sources


# def create_data_source(source_type: str = 'jsonl', **kwargs) -> DataSource:
#     """Factory function to create data source instances."""
#     if source_type == 'jsonl':
#         return JSONLDataSource(**kwargs)
#     elif source_type == 'sqlite':
#         return SQLiteDataSource(**kwargs)
#     else:
#         raise ValueError(f"Unsupported data source type: {source_type}")
