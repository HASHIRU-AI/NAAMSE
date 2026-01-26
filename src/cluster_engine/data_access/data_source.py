from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class DataSource(ABC):
    """BaseClass for data sources that can provide prompts and embeddings."""

    lookup_file: str = 'cluster_lookup_table.json'
    default_source: str = 'NAAMSE_mutation'

    @abstractmethod
    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources."""
        ...

    @abstractmethod
    def check_prompt_exists(self, prompt: str) -> bool:
        """Check if a prompt already exists in the data source."""
        ...

    @abstractmethod
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings for all prompts."""
        ...

    @abstractmethod
    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get cluster metadata for all prompts."""
        ...

    @abstractmethod
    def save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings."""
        ...

    @abstractmethod
    def add_prompt(self, prompt: str, source: str, cluster_info: Optional[Dict[str, Any]] = None) -> None:
        """Add a new prompt to the data source."""
        ...

    @abstractmethod
    def get_prompts_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get all prompts belonging to a specific cluster."""
        ...

    @abstractmethod
    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None, seed=None) -> List[Dict[str, Any]]:
        """Find the n nearest prompts to a given query prompt."""
        ...

    @abstractmethod
    def add_prompt_to_clusters(self, new_prompt: str, device: str = None) -> Dict[str, Any]:
        """Add a new prompt to the existing cluster structure without re-clustering."""
        ...

    @abstractmethod
    def get_random_prompt(self, seed=None) -> Dict[str, Any]:
        """Get a random prompt from the corpus."""
        ...

    @abstractmethod
    def get_cluster_id_for_prompt(self, prompt: List[str], device: str = None) -> Optional[str]:
        """Find the cluster_id for the given prompt."""
        ...

    @abstractmethod
    def get_human_readable_cluster_info(self, cluster_id: str) -> Optional[Dict[str, str]]:
        """Get human-readable label and description for a cluster."""
        ...
