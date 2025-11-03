from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
import numpy as np
import os


class DataSource(Protocol):
    """Protocol for data sources that can provide prompts and embeddings."""

    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources."""
        ...

    def get_embeddings(self) -> np.ndarray:
        """Get embeddings for all prompts."""
        ...

    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get cluster metadata for all prompts."""
        ...

    def save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings."""
        ...

    def add_prompt(self, prompt: str, source: str, cluster_info: Optional[Dict[str, Any]] = None) -> None:
        """Add a new prompt to the data source."""
        ...

    def get_prompts_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get all prompts belonging to a specific cluster."""
        ...


class JSONLDataSource:
    """Data source implementation for JSONL files."""

    def __init__(self, corpus_file: str = 'jailbreak_corpus.jsonl', embeddings_file: str = 'embeddings.npy'):
        # Use script-relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.corpus_file = os.path.join(script_dir, corpus_file)
        self.embeddings_file = os.path.join(script_dir, embeddings_file)

    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources from JSONL file."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_file}")

        prompts = []
        sources = []

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data['messages'][0]['content'])
                sources.append(data['source'])

        return prompts, sources

    def get_embeddings(self) -> np.ndarray:
        """Get embeddings from numpy file."""
        import os

        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        return np.load(self.embeddings_file)

    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get cluster metadata for all prompts."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_file}")

        cluster_info = []

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                cluster_data = {}
                if 'cluster_id' in data:
                    cluster_data['cluster_id'] = data['cluster_id']
                if 'cluster_label' in data:
                    cluster_data['cluster_label'] = data['cluster_label']
                if 'centroid_coord' in data:
                    cluster_data['centroid_coord'] = data['centroid_coord']
                cluster_info.append(cluster_data)

        return cluster_info

    def save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings to numpy file."""
        np.save(self.embeddings_file, embeddings)

    def add_prompt(self, prompt: str, source: str, cluster_info: Optional[Dict[str, Any]] = None) -> None:
        """Add a new prompt to the JSONL file."""
        import json

        new_entry = {
            "source": source,
            "messages": [{"role": "user", "content": prompt}]
        }

        if cluster_info:
            new_entry.update(cluster_info)

        with open(self.corpus_file, 'a', encoding='utf-8') as f:
            json.dump(new_entry, f)
            f.write('\n')

    def get_prompts_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get all prompts belonging to a specific cluster."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_file}")

        cluster_prompts = []

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Check if this entry has cluster_id metadata and matches the requested cluster
                if 'cluster_id' in data and data['cluster_id'] == cluster_id:
                    cluster_prompts.append({
                        'prompt': data['messages'][0]['content'],
                        'source': data['source'],
                        'cluster_id': data['cluster_id'],
                        'cluster_label': data.get('cluster_label', 'Unknown'),
                        'centroid_coord': data.get('centroid_coord', None)
                    })

        return cluster_prompts


# Factory function to create data sources
def create_data_source(source_type: str = 'jsonl', **kwargs) -> DataSource:
    """Factory function to create data source instances."""
    if source_type == 'jsonl':
        return JSONLDataSource(**kwargs)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")