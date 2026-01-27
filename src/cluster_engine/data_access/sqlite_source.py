from typing import List, Dict, Any, Optional
import numpy as np
import os
import random
import torch

from src.cluster_engine.data_access.data_source import DataSource


def get_project_root() -> str:
    """Get the project root directory (parent of cluster_engine)."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return project_root


class SQLiteDataSource(DataSource):
    """Data source implementation for SQLite database."""

    lookup_file: str = 'cluster_lookup_table.json'

    def __init__(self, db_file: str = 'src/cluster_engine/data_access/adversarial/naamse.db',
                 centroids_file: str = 'src/cluster_engine/data_access/adversarial/centroids.pkl',
                 lookup_file: str = 'src/cluster_engine/data_access/adversarial/cluster_lookup_table.json',
                 default_source: str = 'NAAMSE_mutation'):
        """Initialize SQLite data source.

        Args:
            db_file: Path to SQLite database file (relative to project root if not absolute)
            centroids_file: Path to centroids file (relative to cluster_engine directory if not absolute)
            lookup_file: Path to cluster lookup table JSON file
            default_source: Default source identifier for new prompts
        """

        project_root = get_project_root()

        # Make db_file path absolute if relative
        if not os.path.isabs(db_file):
            # Use project root (parent of cluster_engine) for DB files
            db_file = os.path.join(project_root, db_file)

        # Make centroids_file path absolute if relative
        if not os.path.isabs(centroids_file):
            centroids_file = os.path.join(project_root, centroids_file)

        # make the path to lookup_file absolute if relative (paths are calculated from project root)
        if not os.path.isabs(lookup_file):
            lookup_file = os.path.join(project_root, lookup_file)

        self._db_file = db_file
        self.centeroids_file = centroids_file
        self.lookup_file = lookup_file
        self.default_source = default_source
        self._embeddings_cache = None
        self._model_cache = {}

        # Verify database exists
        if not os.path.exists(self._db_file):
            raise FileNotFoundError(
                f"Database file not found: {self._db_file}")

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

        cursor.execute(
            "SELECT 1 FROM prompts WHERE user_content = ? LIMIT 1", (prompt,))
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
            # Unpack blob as array of floats (stored as raw bytes)
            embedding = np.frombuffer(blob, dtype=np.float32)
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

    def add_prompt(self, prompt: str, source: str, cluster_info: Optional[Dict[str, Any]] = None) -> int:
        """Add a new prompt to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cluster_id = cluster_info.get('cluster_id', '') if cluster_info else ''
        cluster_label = cluster_info.get(
            'cluster_label', '') if cluster_info else ''

        # Calculate metadata
        content_length = len(prompt)
        word_count = len(prompt.split())
        cluster_depth = len(cluster_id.split('/')) if cluster_id else 0

        cursor.execute("""
            INSERT INTO prompts 
            (source, cluster_id, cluster_label, user_content, content_length, word_count, cluster_depth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, cluster_id, cluster_label, prompt, content_length, word_count, cluster_depth))

        # Get the inserted ID
        new_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return new_id

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
                centroid_coord = np.frombuffer(blob, dtype=np.float32).tolist()

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
            self._model_cache[cache_key] = SentenceTransformer(
                'all-MiniLM-L6-v2', device=device)

        return self._model_cache[cache_key]

    def find_nearest_prompts(self, query_prompt: str, n: int = 1, device: str = None, seed=None) -> List[Dict[str, Any]]:
        """Find n random prompts from the parent cluster of the given query prompt."""

        # print(f"\n[DEBUG SQLITE] Query prompt: {query_prompt[:60]}...")

        # Use seeded random for reproducible sampling if seed provided
        rng = random.Random(seed) if seed is not None else random

        conn = self._get_connection()
        cursor = conn.cursor()

        # Find the cluster_id of the query_prompt
        cursor.execute(
            "SELECT id, cluster_id FROM prompts WHERE user_content = ?", (query_prompt,))
        row = cursor.fetchone()

        if not row:
            # If not found, find the nearest prompt's cluster
            # print(f"[DEBUG SQLITE] Prompt NOT found in DB, using embedding search...")
            embeddings = self.get_embeddings()
            if device is None:
                device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = self._get_cached_model(device)
            query_embedding = model.encode(
                [query_prompt], normalize_embeddings=True)[0]
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / \
                np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
            top_index = np.argmax(similarities)
            # print(f"[DEBUG SQLITE] Nearest prompt index: {top_index}, similarity: {similarities[top_index]:.4f}")

            # Get the actual prompt_id from centroids table at this index
            cursor.execute("""
                SELECT prompt_id FROM centroids ORDER BY prompt_id LIMIT 1 OFFSET ?
            """, (int(top_index),))
            centroid_row = cursor.fetchone()
            if centroid_row:
                prompt_id = centroid_row[0]
                cursor.execute(
                    "SELECT cluster_id FROM prompts WHERE id = ?", (prompt_id,))
                row2 = cursor.fetchone()
                if row2:
                    cluster_id = row2[0]
                # print(f"[DEBUG SQLITE] Nearest prompt cluster: {cluster_id}")
            else:
                conn.close()
                # print(f"[DEBUG SQLITE] No cluster found for nearest prompt, returning empty list")
                return []
        else:
            prompt_id, cluster_id = row
            # print(f"[DEBUG SQLITE] Prompt found in DB! ID: {prompt_id}, Cluster: {cluster_id}")

        if not cluster_id:
            conn.close()
            # print(f"[DEBUG SQLITE] No cluster_id found, returning empty list")
            return []

        parts = cluster_id.split('/')
        # print(f"[DEBUG SQLITE] Original cluster parts: {parts} (depth: {len(parts)})")

        if len(parts) <= 1:
            where_clause = "p.cluster_id = ?"
            param = cluster_id
            # print(f"[DEBUG SQLITE] Top-level cluster, staying at: {param}")
        else:
            parent = '/'.join(parts[:-1])
            where_clause = "p.cluster_id LIKE ?"
            param = parent + '/%'
            # print(f"[DEBUG SQLITE] Going up to parent cluster: {parent}")

        # Get all prompts from the parent cluster (excluding the query prompt)
        # print(f"[DEBUG SQLITE] Querying with WHERE clause: {where_clause}, param: {param}")
        cursor.execute(f"""
            SELECT p.id, p.user_content, p.source, p.cluster_id, p.cluster_label,
                   c.embedding_vector, c.dimensions
            FROM prompts p
            LEFT JOIN centroids c ON p.id = c.prompt_id
            WHERE {where_clause} AND p.user_content != ?
            ORDER BY p.id
        """, (param, query_prompt))

        all_rows = cursor.fetchall()
        # print(f"[DEBUG SQLITE] Found {len(all_rows)} results from parent cluster")

        # Use seeded random to select n prompts
        if len(all_rows) <= n:
            selected_rows = all_rows
        else:
            selected_rows = rng.sample(all_rows, n)

        results = []
        for row in selected_rows:
            result = {
                'prompt': row[1],
                'source': row[2],
                'similarity': None,
                'index': row[0] - 1
            }
            if row[3]:
                result['cluster_id'] = row[3]
            if row[4]:
                result['cluster_label'] = row[4]
            if row[5] is not None:
                blob = row[5]
                dimensions = row[6]
                result['centroid_coord'] = np.frombuffer(
                    blob, dtype=np.float32).tolist()
            results.append(result)

        conn.close()
        return results

    def add_prompt_to_clusters(self, new_prompt: str, device: str = None) -> Dict[str, Any]:
        """Add a new prompt to the existing cluster structure without re-clustering."""
        import pickle
        import torch

        # Check for duplicates
        if self.check_prompt_exists(new_prompt):
            print(f"⚠️  Skipping duplicate prompt: {new_prompt[:50]}...")
            return {
                'prompt': new_prompt,
                'source': self.default_source,
                'cluster_id': None,
                'cluster_label': None,
                'centroid_coord': None,
                'distance_to_centroid': None,
                'embedding_index': None
            }

        # Load centroids - try pickle file first, then compute from database
        centroids = None
        if os.path.exists(self.centeroids_file):
            try:
                with open(self.centeroids_file, 'rb') as f:
                    centroids = pickle.load(f)
                print(f"Loaded {len(centroids)} centroids from pickle file")
            except Exception as e:
                print(
                    f"Warning: Could not load centroids from pickle file: {e}")

        # If pickle file not available or failed to load, compute centroids from database
        if centroids is None:
            print("Computing centroids from database...")
            centroids = self._compute_cluster_centroids_from_db()
            print(f"Computed {len(centroids)} centroids from database")

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
        new_prompt_id = self.add_prompt(
            new_prompt, self.default_source, cluster_info)

        # Update embeddings - add the new embedding
        import struct
        conn = self._get_connection()
        cursor = conn.cursor()

        # Store the embedding
        blob = struct.pack(f'{len(new_embedding)}f', *new_embedding)
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
            'source': self.default_source,
            'cluster_id': cluster_path,
            'cluster_label': cluster_label,
            'centroid_coord': nearest_centroid.tolist(),
            'distance_to_centroid': float(nearest_distance),
            'embedding_index': new_prompt_id - 1
        }

        print("✅ Prompt added successfully!")
        return result

    def get_random_prompt(self, seed=None) -> Dict[str, Any]:
        """Get a random prompt from the corpus."""

        conn = self._get_connection()
        cursor = conn.cursor()

        # Get total count of prompts
        cursor.execute("SELECT COUNT(*) FROM prompts")
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            raise ValueError("No prompts found in the database")

        # Use seeded random for reproducible sampling if seed provided
        rng = random.Random(seed) if seed is not None else random
        random_offset = rng.randint(0, count - 1)

        cursor.execute("""
            SELECT id, user_content, source, cluster_id, cluster_label
            FROM prompts
            ORDER BY id
            LIMIT 1 OFFSET ?
        """, (random_offset,))

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
        # remember to pass seed for reproducibility
        nearest = self.find_nearest_prompts(prompt_text, n=1, device=device)
        if not nearest:
            return None

        nearest_data = nearest[0]
        cluster_id = nearest_data.get('cluster_id')
        return cluster_id

    def get_human_readable_cluster_info(self, cluster_id: str) -> Optional[Dict[str, str]]:
        """Get human-readable label and description for a cluster."""
        import json
        import pickle

        print(f"Getting description for cluster_id: {cluster_id}")

        # Make lookup_file path relative to script directory if not absolute
        if not os.path.isabs(self.lookup_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lookup_file = os.path.join(script_dir, self.lookup_file)
        else:
            lookup_file = self.lookup_file

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
        cursor.execute(
            "SELECT AVG(content_length), AVG(word_count) FROM prompts")
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

    def _compute_cluster_centroids_from_db(self) -> Dict[str, np.ndarray]:
        """Compute cluster centroids from embeddings stored in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all embeddings with their cluster_ids
        cursor.execute("""
            SELECT p.cluster_id, c.embedding_vector, c.dimensions
            FROM prompts p
            JOIN centroids c ON p.id = c.prompt_id
            WHERE p.cluster_id IS NOT NULL AND p.cluster_id != ''
            ORDER BY p.cluster_id
        """)

        cluster_embeddings = {}
        for row in cursor.fetchall():
            cluster_id = row[0]
            blob = row[1]
            dimensions = row[2]

            # Unpack the embedding (stored as raw bytes, assuming float32)
            embedding = np.frombuffer(blob, dtype=np.float32)

            if cluster_id not in cluster_embeddings:
                cluster_embeddings[cluster_id] = []
            cluster_embeddings[cluster_id].append(embedding)

        conn.close()

        # Compute centroids (mean of embeddings for each cluster)
        centroids = {}
        for cluster_id, embeddings in cluster_embeddings.items():
            if embeddings:  # Make sure we have at least one embedding
                centroid = np.mean(embeddings, axis=0)
                centroids[cluster_id] = centroid

        return centroids
