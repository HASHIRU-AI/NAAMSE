from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
import numpy as np
import os


class DataSource(Protocol):
    """Protocol for data sources that can provide prompts and embeddings."""

    @property
    def corpus_file(self) -> str:
        """Get the path to the corpus file."""
        ...

    @property
    def embeddings_file(self) -> str:
        """Get the path to the embeddings file."""
        ...

    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources."""
        ...

    def check_prompt_exists(self, prompt: str) -> bool:
        """Check if a prompt already exists in the data source."""
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

    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None) -> List[Dict[str, Any]]:
        """Find the n nearest prompts to a given query prompt."""
        ...

    def add_prompt_to_clusters(self, new_prompt: str, source: str, centroids_file: str, device: str = None) -> Dict[str, Any]:
        """Add a new prompt to the existing cluster structure without re-clustering."""
        ...

    def get_random_prompt(self) -> Dict[str, Any]:
        """Get a random prompt from the corpus."""
        ...

    def get_cluster_id_for_prompt(self, prompt: List[str], device: str = None) -> Optional[str]:
        """Find the cluster_id for the given prompt."""
        ...

    def get_human_readable_cluster_info(self, cluster_id: str, lookup_file: str) -> Optional[Dict[str, str]]:
        """Get human-readable label and description for a cluster."""
        ...


class JSONLDataSource:
    """Data source implementation for JSONL files."""

    def __init__(self, corpus_file: str = 'jailbreak_corpus.jsonl', embeddings_file: str = 'embeddings.npy'):
        # Use script-relative paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.corpus_file = os.path.join(script_dir, corpus_file)
        self.embeddings_file = os.path.join(script_dir, embeddings_file)
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
            self._model_cache[cache_key] = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        return self._model_cache[cache_key]

    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None) -> List[Dict[str, Any]]:
        """Find the n nearest prompts to a given query prompt."""
        import json
        import torch

        # Load embeddings
        embeddings = self.get_embeddings()

        # Determine device
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Encode query prompt using cached model
        model = self._get_cached_model(device)
        query_embedding = model.encode([query_prompt], normalize_embeddings=True)[0]

        # Use brute force for nearest neighbor search
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top n indices
        top_n_indices = np.argsort(similarities)[::-1][:n]

        # Load only the required prompts
        top_indices_set = set(top_n_indices)
        selected_data = {}

        with open(self.corpus_file, 'r') as f:
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
            if 'cluster_id' in data:
                result['cluster_id'] = data['cluster_id']
            if 'cluster_label' in data:
                result['cluster_label'] = data['cluster_label']
            if 'centroid_coord' in data:
                result['centroid_coord'] = data['centroid_coord']
            results.append(result)

        return results

    def add_prompt_to_clusters(self, new_prompt: str, source: str, centroids_file: str, device: str = None) -> Dict[str, Any]:
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

        print(f"Nearest cluster: {nearest_cluster_id} (distance: {nearest_distance:.4f})")

        # Get cluster label
        cluster_path = nearest_cluster_id
        cluster_label = f"Cluster_{nearest_cluster_id.replace('/', '_')}"

        try:
            cluster_prompts = self.get_prompts_by_cluster(nearest_cluster_id)
            if cluster_prompts:
                cluster_label = cluster_prompts[0].get('cluster_label', cluster_label)
        except Exception as e:
            print(f"Warning: Could not retrieve cluster label from data source: {e}")

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

    def get_random_prompt(self) -> Dict[str, Any]:
        """Get a random prompt from the corpus using reservoir sampling."""
        import json
        import random

        if not os.path.exists(self.corpus_file):
            raise ValueError("No prompts found in the corpus")

        # Reservoir sampling
        selected_line = None
        selected_index = -1
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for index, line in enumerate(f, start=1):
                if random.randrange(index) == 0:
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


class SQLiteDataSource:
    """Data source implementation for SQLite database."""

    def __init__(self, db_file: str = 'naamse.db'):
        """Initialize SQLite data source.
        
        Args:
            db_file: Path to SQLite database file (relative to project root if not absolute)
        """
        import sqlite3
        
        # Make db_file path absolute if relative
        if not os.path.isabs(db_file):
            # Use project root (parent of cluster_engine) for DB files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            db_file = os.path.join(project_root, db_file)
        
        self._db_file = db_file
        self._embeddings_cache = None
        self._model_cache = {}
        
        # Verify database exists
        if not os.path.exists(self._db_file):
            raise FileNotFoundError(f"Database file not found: {self._db_file}")
    
    @property
    def corpus_file(self) -> str:
        """Get the path to the corpus file (DB file for SQLite)."""
        return self._db_file
    
    @property
    def embeddings_file(self) -> str:
        """Get the path to the embeddings file (DB file for SQLite)."""
        return self._db_file
    
    def _get_connection(self):
        """Get a database connection."""
        import sqlite3
        return sqlite3.connect(self._db_file)
    
    def get_prompts_and_sources(self) -> tuple[List[str], List[str]]:
        """Get all prompts and their sources from SQLite database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_content, source FROM prompts ORDER BY id")
        rows = cursor.fetchall()
        
        prompts = [row[0] for row in rows]
        sources = [row[1] for row in rows]
        
        conn.close()
        return prompts, sources
    
    def check_prompt_exists(self, prompt: str) -> bool:
        """Check if a prompt already exists in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM prompts WHERE user_content = ? LIMIT 1", (prompt,))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
    
    def get_embeddings(self) -> np.ndarray:
        """Get embeddings from database (with caching)."""
        import struct
        
        # Return cached embeddings if available
        if self._embeddings_cache is not None:
            return self._embeddings_cache
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get embeddings ordered by prompt_id to maintain correspondence
        cursor.execute("""
            SELECT embedding_vector, dimensions 
            FROM centroids 
            ORDER BY prompt_id
        """)
        
        embeddings = []
        for row in cursor.fetchall():
            blob = row[0]
            dimensions = row[1]
            # Unpack blob as array of floats (assuming double precision - 8 bytes each)
            embedding = struct.unpack(f'{dimensions}d', blob)
            embeddings.append(embedding)
        
        conn.close()
        
        # Convert to numpy array and cache
        self._embeddings_cache = np.array(embeddings, dtype=np.float64)
        return self._embeddings_cache
    
    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get cluster metadata for all prompts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT cluster_id, cluster_label 
            FROM prompts 
            ORDER BY id
        """)
        
        cluster_info = []
        for row in cursor.fetchall():
            cluster_info.append({
                'cluster_id': row[0],
                'cluster_label': row[1]
            })
        
        conn.close()
        return cluster_info
    
    def save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings to database and update cache."""
        pass
    
    def add_prompt(self, prompt: str, source: str, cluster_info: Optional[Dict[str, Any]] = None) -> None:
        """Add a new prompt to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cluster_id = cluster_info.get('cluster_id', '') if cluster_info else ''
        cluster_label = cluster_info.get('cluster_label', '') if cluster_info else ''
        
        # Calculate metadata
        content_length = len(prompt)
        word_count = len(prompt.split())
        cluster_depth = len(cluster_id.split('/')) if cluster_id else 0
        
        cursor.execute("""
            INSERT INTO prompts 
            (source, cluster_id, cluster_label, user_content, content_length, word_count, cluster_depth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, cluster_id, cluster_label, prompt, content_length, word_count, cluster_depth))
        
        conn.commit()
        conn.close()
    
    def get_prompts_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Get all prompts belonging to a specific cluster."""
        import struct
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.id, p.user_content, p.source, p.cluster_id, p.cluster_label,
                   c.embedding_vector, c.dimensions
            FROM prompts p
            LEFT JOIN centroids c ON p.id = c.prompt_id
            WHERE p.cluster_id = ?
            ORDER BY p.id
        """, (cluster_id,))
        
        cluster_prompts = []
        for row in cursor.fetchall():
            centroid_coord = None
            if row[5] is not None:  # embedding_vector exists
                blob = row[5]
                dimensions = row[6]
                centroid_coord = list(struct.unpack(f'{dimensions}d', blob))
            
            cluster_prompts.append({
                'prompt': row[1],
                'source': row[2],
                'cluster_id': row[3],
                'cluster_label': row[4],
                'centroid_coord': centroid_coord
            })
        
        conn.close()
        return cluster_prompts
    
    def _get_cached_model(self, device: str = None):
        """Get a cached SentenceTransformer model to avoid reloading on every call."""
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        cache_key = f'all-MiniLM-L6-v2_{device}'

        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        return self._model_cache[cache_key]
    
    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None) -> List[Dict[str, Any]]:
        """Find the n nearest prompts to a given query prompt."""
        import torch
        
        # Load embeddings
        embeddings = self.get_embeddings()
        
        # Determine device
        if device is None:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Encode query prompt using cached model
        model = self._get_cached_model(device)
        query_embedding = model.encode([query_prompt], normalize_embeddings=True)[0]
        
        # Use brute force for nearest neighbor search
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)
        top_n_indices = np.argsort(similarities)[::-1][:n]
        similarities = similarities[top_n_indices]
        
        # Fetch the corresponding prompts from database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build results
        results = []
        for i, idx in enumerate(top_n_indices):
            # prompt_id is idx + 1 (1-indexed in DB)
            prompt_id = int(idx) + 1
            
            cursor.execute("""
                SELECT p.user_content, p.source, p.cluster_id, p.cluster_label,
                       c.embedding_vector, c.dimensions
                FROM prompts p
                LEFT JOIN centroids c ON p.id = c.prompt_id
                WHERE p.id = ?
            """, (prompt_id,))
            
            row = cursor.fetchone()
            if row:
                result = {
                    'prompt': row[0],
                    'source': row[1],
                    'similarity': float(similarities[i]),
                    'index': int(idx)
                }
                if row[2]:  # cluster_id
                    result['cluster_id'] = row[2]
                if row[3]:  # cluster_label
                    result['cluster_label'] = row[3]
                if row[4] is not None:  # embedding_vector exists
                    import struct
                    blob = row[4]
                    dimensions = row[5]
                    result['centroid_coord'] = list(struct.unpack(f'{dimensions}d', blob))
                results.append(result)
        
        conn.close()
        return results
    
    def add_prompt_to_clusters(self, new_prompt: str, source: str, centroids_file: str, device: str = None) -> Dict[str, Any]:
        """Add a new prompt to the existing cluster structure without re-clustering."""
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
        
        print(f"Nearest cluster: {nearest_cluster_id} (distance: {nearest_distance:.4f})")
        
        # Get cluster label
        cluster_path = nearest_cluster_id
        cluster_label = f"Cluster_{nearest_cluster_id.replace('/', '_')}"
        
        try:
            cluster_prompts = self.get_prompts_by_cluster(nearest_cluster_id)
            if cluster_prompts:
                cluster_label = cluster_prompts[0].get('cluster_label', cluster_label)
        except Exception as e:
            print(f"Warning: Could not retrieve cluster label from data source: {e}")
        
        # Create cluster info
        cluster_info = {
            "cluster_id": cluster_path,
            "cluster_label": cluster_label,
            "centroid_coord": nearest_centroid.tolist()
        }
        
        # Add to data source
        self.add_prompt(new_prompt, source, cluster_info)
        
        # Update embeddings - add the new embedding
        import struct
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get the ID of the newly inserted prompt
        cursor.execute("SELECT last_insert_rowid()")
        new_prompt_id = cursor.fetchone()[0]
        
        # Store the embedding
        blob = struct.pack(f'{len(new_embedding)}d', *new_embedding)
        cursor.execute("""
            INSERT INTO centroids (prompt_id, embedding_vector, dimensions)
            VALUES (?, ?, ?)
        """, (new_prompt_id, blob, len(new_embedding)))
        
        conn.commit()
        conn.close()
        
        # Invalidate embeddings cache
        self._embeddings_cache = None
        
        result = {
            'prompt': new_prompt,
            'source': source,
            'cluster_id': cluster_path,
            'cluster_label': cluster_label,
            'centroid_coord': nearest_centroid.tolist(),
            'distance_to_centroid': float(nearest_distance),
            'embedding_index': new_prompt_id - 1
        }
        
        print("✅ Prompt added successfully!")
        return result
    
    def get_random_prompt(self) -> Dict[str, Any]:
        """Get a random prompt from the corpus."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Use SQLite's RANDOM() for efficient random sampling
        cursor.execute("""
            SELECT id, user_content, source, cluster_id, cluster_label
            FROM prompts
            ORDER BY RANDOM()
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError("No prompts found in the database")
        
        result = {
            'prompt': row[1],
            'source': row[2],
            'index': row[0] - 1  # Convert to 0-indexed
        }
        
        if row[3]:  # cluster_id
            result['cluster_id'] = row[3]
        if row[4]:  # cluster_label
            result['cluster_label'] = row[4]
        
        return result
    
    def get_cluster_id_for_prompt(self, prompt: List[str], device: str = None) -> Optional[str]:
        """Find the cluster_id for the given prompt."""
        prompt_text = prompt[0] if len(prompt) > 0 else ""
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
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get statistics about clusters in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total prompts
        cursor.execute("SELECT COUNT(*) FROM prompts")
        total_prompts = cursor.fetchone()[0]
        
        # Unique clusters
        cursor.execute("SELECT COUNT(DISTINCT cluster_id) FROM prompts")
        unique_clusters = cursor.fetchone()[0]
        
        # Average content length and word count
        cursor.execute("SELECT AVG(content_length), AVG(word_count) FROM prompts")
        avg_content_length, avg_word_count = cursor.fetchone()
        
        # Top clusters by prompt count
        cursor.execute("""
            SELECT cluster_id, cluster_label, COUNT(*) as prompt_count
            FROM prompts
            GROUP BY cluster_id
            ORDER BY prompt_count DESC
            LIMIT 10
        """)
        
        top_clusters = []
        for row in cursor.fetchall():
            top_clusters.append({
                'cluster_id': row[0],
                'cluster_label': row[1],
                'prompt_count': row[2]
            })
        
        conn.close()
        
        return {
            'total_prompts': total_prompts,
            'unique_clusters': unique_clusters,
            'avg_content_length': avg_content_length,
            'avg_word_count': avg_word_count,
            'top_clusters': top_clusters
        }


# Factory function to create data sources
def create_data_source(source_type: str = 'jsonl', **kwargs) -> DataSource:
    """Factory function to create data source instances."""
    if source_type == 'jsonl':
        return JSONLDataSource(**kwargs)
    elif source_type == 'sqlite':
        return SQLiteDataSource(**kwargs)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")
