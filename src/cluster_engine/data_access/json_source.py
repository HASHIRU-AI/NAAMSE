from typing import List, Dict, Any, Optional
import numpy as np
import os
import json
import random
import torch

from src.cluster_engine.data_access.data_source import DataSource

class JSONLDataSource(DataSource):
    """Data source implementation for JSONL files."""

    def __init__(self, corpus_file: str = 'jailbreak_corpus.jsonl', embeddings_file: str = 'embeddings.npy', centroids_file: str = 'centroids.pkl'):
        # Use script-relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.corpus_file = os.path.join(script_dir, corpus_file)
        self.embeddings_file = os.path.join(script_dir, embeddings_file)
        self.centeroids_file = os.path.join(script_dir, centroids_file)
        self._embeddings_cache = None  # Cache for embeddings to avoid repeated disk reads
        self._model_cache = {}  # Cache for SentenceTransformer models

    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources from JSONL file."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(
                f"Corpus file not found: {self.corpus_file}")

        prompts = []
        sources = []

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data['messages'][0]['content'])
                sources.append(data['source'])

        return prompts, sources

    def check_prompt_exists(self, prompt: str) -> bool:
        """Check if a prompt already exists in the JSONL file."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(
                f"Corpus file not found: {self.corpus_file}")

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['messages'][0]['content'] == prompt:
                    return True

        return False

    def get_embeddings(self) -> np.ndarray:
        """Get embeddings from numpy file (with caching)."""
        import os

        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_file}")

        # Return cached embeddings if available
        if self._embeddings_cache is not None:
            return self._embeddings_cache

        # Load and cache embeddings
        self._embeddings_cache = np.load(self.embeddings_file)
        return self._embeddings_cache

    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get cluster metadata for all prompts."""
        import json
        import os

        if not os.path.exists(self.corpus_file):
            raise FileNotFoundError(
                f"Corpus file not found: {self.corpus_file}")

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
        """Save embeddings to numpy file and update cache."""
        np.save(self.embeddings_file, embeddings)
        self._embeddings_cache = embeddings  # Update cache

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
            raise FileNotFoundError(
                f"Corpus file not found: {self.corpus_file}")

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

    def _get_cached_model(self, device: str = None):
        """Get a cached SentenceTransformer model to avoid reloading on every call."""
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        cache_key = f'all-MiniLM-L6-v2_{device}'

        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = SentenceTransformer(
                'all-MiniLM-L6-v2', device=device)

        return self._model_cache[cache_key]

    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None, seed=None) -> List[Dict[str, Any]]:
        """Find n random prompts from the parent cluster of the given query prompt."""

        print(f"\n[DEBUG JSONL] Query prompt: {query_prompt[:60]}...")

        # Load embeddings
        embeddings = self.get_embeddings()

        # Determine device
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Use seeded random for reproducible sampling if seed provided
        rng = random.Random(seed) if seed is not None else random

        # First, find the cluster_id of the query_prompt
        cluster_id = None
        prompt_found = False

        with open(self.corpus_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['messages'][0]['content'] == query_prompt:
                    cluster_id = data.get('cluster_id')
                    prompt_found = True
                    # print(f"[DEBUG JSONL] Prompt found in corpus! Cluster: {cluster_id}")
                    break

        # If prompt not found, find nearest via embeddings
        if not prompt_found:
            # print(f"[DEBUG JSONL] Prompt NOT found, using embedding search...")
            model = self._get_cached_model(device)
            query_embedding = model.encode(
                [query_prompt], normalize_embeddings=True)[0]
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
            top_index = np.argmax(similarities)
            # print(f"[DEBUG JSONL] Nearest prompt index: {top_index}, similarity: {similarities[top_index]:.4f}")

            # Get cluster_id of nearest prompt
            with open(self.corpus_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num == top_index:
                        data = json.loads(line.strip())
                        cluster_id = data.get('cluster_id')
                        # print(f"[DEBUG JSONL] Nearest prompt cluster: {cluster_id}")
                        break

        if not cluster_id:
            # print(f"[DEBUG JSONL] No cluster_id found, returning empty list")
            return []

        # Determine parent cluster
        parts = cluster_id.split('/')
        # print(f"[DEBUG JSONL] Original cluster parts: {parts} (depth: {len(parts)})")

        if len(parts) <= 1:
            parent_prefix = cluster_id
            exact_match = True
            # print(f"[DEBUG JSONL] Top-level cluster, staying at: {parent_prefix}")
        else:
            parent_prefix = '/'.join(parts[:-1])
            exact_match = False
            # print(f"[DEBUG JSONL] Going up to parent cluster: {parent_prefix}")

        # Collect all prompts from parent cluster (excluding query prompt)
        candidate_prompts = []
        with open(self.corpus_file, 'r') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                prompt_cluster = data.get('cluster_id', '')
                prompt_content = data['messages'][0]['content']

                # Skip the query prompt
                if prompt_content == query_prompt:
                    continue

                # Check if prompt belongs to parent cluster
                if exact_match:
                    matches = prompt_cluster == parent_prefix
                else:
                    matches = prompt_cluster.startswith(parent_prefix + '/')

                if matches:
                    candidate_prompts.append({
                        'prompt': prompt_content,
                        'source': data['source'],
                        'similarity': None,
                        'index': line_num,
                        'cluster_id': data.get('cluster_id'),
                        'cluster_label': data.get('cluster_label'),
                        'centroid_coord': data.get('centroid_coord')
                    })

        # print(f"[DEBUG JSONL] Found {len(candidate_prompts)} candidates in parent cluster")

        # Return random n prompts
        if len(candidate_prompts) <= n:
            # print(f"[DEBUG JSONL] Returning all {len(candidate_prompts)} candidates")
            return candidate_prompts

        selected = rng.sample(candidate_prompts, n)
        # print(f"[DEBUG JSONL] Randomly selected {len(selected)} prompts from {len(candidate_prompts)} candidates")
        return selected

    def add_prompt_to_clusters(self, new_prompt: str, source: str, device: str = None) -> Dict[str, Any]:
        """Add a new prompt to the existing cluster structure without re-clustering."""
        import json
        import pickle
        import torch

        # Check for duplicates
        if self.check_prompt_exists(new_prompt):
            print(f"⚠️  Skipping duplicate prompt: {new_prompt[:50]}...")
            return {
                'prompt': new_prompt,
                'source': source,
                'cluster_id': None,
                'cluster_label': None,
                'centroid_coord': None,
                'distance_to_centroid': None,
                'embedding_index': None
            }

        # Load centroids
        if not os.path.exists(self.centeroids_file):
            raise FileNotFoundError(
                f"Centroids file not found: {self.centeroids_file}. Run clustering first.")

        with open(self.centeroids_file, 'rb') as f:
            centroids = pickle.load(f)

        # Determine device
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Embed the new prompt
        print(f"Embedding new prompt on {device}...")
        model = self._get_cached_model(device)
        new_embedding = model.encode([new_prompt])[0]

        # Find nearest centroid
        distances = {}
        for cluster_id, centroid in centroids.items():
            dist = np.linalg.norm(new_embedding - centroid)
            distances[cluster_id] = dist

        nearest_cluster_id = min(distances, key=distances.get)
        nearest_distance = distances[nearest_cluster_id]
        nearest_centroid = centroids[nearest_cluster_id]

        print(
            f"Nearest cluster: {nearest_cluster_id} (distance: {nearest_distance:.4f})")

        # Get cluster label
        cluster_path = nearest_cluster_id
        cluster_label = f"Cluster_{nearest_cluster_id.replace('/', '_')}"

        try:
            cluster_prompts = self.get_prompts_by_cluster(nearest_cluster_id)
            if cluster_prompts:
                cluster_label = cluster_prompts[0].get(
                    'cluster_label', cluster_label)
        except Exception as e:
            print(
                f"Warning: Could not retrieve cluster label from data source: {e}")

        # Create cluster info
        cluster_info = {
            "cluster_id": cluster_path,
            "cluster_label": cluster_label,
            "centroid_coord": nearest_centroid.tolist()
        }

        # Add to data source
        self.add_prompt(new_prompt, source, cluster_info)

        # Update embeddings
        existing_embeddings = self.get_embeddings()
        updated_embeddings = np.vstack([existing_embeddings, new_embedding])
        self.save_embeddings(updated_embeddings)

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

    def get_random_prompt(self, seed=None) -> Dict[str, Any]:
        """Get a random prompt from the corpus using reservoir sampling."""
        if not os.path.exists(self.corpus_file):
            raise ValueError("No prompts found in the corpus")

        # Use seeded random for reproducible sampling if seed provided
        rng = random.Random(seed) if seed is not None else random

        # Reservoir sampling
        selected_line = None
        selected_index = -1
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for index, line in enumerate(f, start=1):
                if rng.randrange(index) == 0:
                    selected_line = line
                    selected_index = index - 1

        if selected_line is None:
            raise ValueError("No prompts found in the corpus")

        selected_data = json.loads(selected_line.strip())

        # Build result
        result = {
            'prompt': selected_data['messages'][0]['content'],
            'source': selected_data['source'],
            'index': selected_index
        }

        if 'cluster_id' in selected_data:
            result['cluster_id'] = selected_data['cluster_id']
        if 'cluster_label' in selected_data:
            result['cluster_label'] = selected_data['cluster_label']
        if 'centroid_coord' in selected_data:
            result['centroid_coord'] = selected_data['centroid_coord']

        return result

    def get_cluster_id_for_prompt(self, prompt: List[str], device: str = None) -> Optional[str]:
        """Find the cluster_id for the given prompt."""
        prompt_text = prompt[0] if len(prompt) > 0 else ""
        # remember to pass seed for reproducibility
        nearest = self.find_nearest_prompts(prompt_text, n=1, device=device)
        if not nearest:
            return None

        nearest_data = nearest[0]
        cluster_id = nearest_data.get('cluster_id')
        return cluster_id

    def get_human_readable_cluster_info(self, cluster_id: str, lookup_file: str) -> Optional[Dict[str, str]]:
        """Get human-readable label and description for a cluster."""
        import json
        import pickle

        print(f"Getting description for cluster_id: {cluster_id}")

        # Make lookup_file path relative to script directory if not absolute
        if not os.path.isabs(lookup_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lookup_file = os.path.join(script_dir, lookup_file)

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
            centroids_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'centroids.pkl')
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

