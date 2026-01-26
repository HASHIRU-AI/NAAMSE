#!/usr/bin/env python3
"""
SQLite-based Clustering Workflow for NAAMSE Databases

Adapts the LangGraph clustering workflow to work with SQLite databases
instead of JSONL files. Processes both adversarial (naamse.db) and benign
(naamse_benign.db) datasets.
"""

import os
import sqlite3
import numpy as np
import torch
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from tqdm import tqdm

# Import node functions (we'll adapt them)
from src.cluster_engine.nodes.embed_prompts import embed_prompts
from src.cluster_engine.nodes.optimize_kmeans import optimize_kmeans
from src.cluster_engine.nodes.cluster_data import cluster_data
from src.cluster_engine.nodes.hierarchical_clustering import hierarchical_clustering
from src.cluster_engine.nodes.label_clusters import label_clusters
from src.cluster_engine.nodes.types import ClusteringState


def load_data_sqlite(db_path: str) -> Dict[str, Any]:
    """Load prompts and sources from SQLite database."""
    print(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get prompts with their IDs for later updating
    cursor.execute("SELECT id, user_content, source FROM prompts")
    rows = cursor.fetchall()

    prompts = []
    sources = []
    prompt_ids = []

    for row in rows:
        prompt_ids.append(row[0])
        prompts.append(row[1])
        sources.append(row[2])

    conn.close()

    print(f"Loaded {len(prompts)} prompts from {db_path}")

    return {
        'prompts': prompts,
        'sources': sources,
        'prompt_ids': prompt_ids,  # Keep track of IDs for updating DB
        'embeddings_file': f'{os.path.splitext(db_path)[0]}_embeddings.npy',
        'checkpoint_dir': f'{os.path.splitext(db_path)[0]}_checkpoints',
        'db_path': db_path,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }


def save_results_sqlite(state: ClusteringState) -> ClusteringState:
    """Update SQLite database with cluster metadata and embeddings."""
    print("Updating SQLite database with cluster metadata and embeddings...")

    db_path = state['db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create mapping from prompt to cluster info
    prompt_to_cluster = {}
    for i, fc in enumerate(state['final_clusters']):
        cluster_id = fc['path']
        cluster_label = state['cluster_labels'][i]
        # Calculate cluster depth from path
        cluster_depth = len(cluster_id.split('/')) - 1 if '/' in cluster_id else 0

        for prompt in fc['prompts']:
            prompt_to_cluster[prompt] = {
                'cluster_id': cluster_id,
                'cluster_label': cluster_label,
                'cluster_depth': cluster_depth
            }

    # Create mapping from prompt to id for faster updates
    prompt_to_id = {prompt: pid for prompt, pid in zip(state['prompts'], state['prompt_ids'])}

    # Update prompts table with cluster info
    updated_count = 0
    for prompt, cluster_info in tqdm(prompt_to_cluster.items(), desc="Updating prompts"):
        pid = prompt_to_id[prompt]
        cursor.execute("""
            UPDATE prompts
            SET cluster_id = ?, cluster_label = ?, cluster_depth = ?
            WHERE id = ?
        """, (
            cluster_info['cluster_id'],
            cluster_info['cluster_label'],
            cluster_info['cluster_depth'],
            pid
        ))
        updated_count += cursor.rowcount

    # Load embeddings and save to centroids table
    embeddings_file = state['embeddings_file']
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        
        # Clear existing centroids
        cursor.execute("DELETE FROM centroids")
        
        # Insert new centroids (embeddings per prompt)
        for i, prompt_id in tqdm(enumerate(state['prompt_ids']), desc="Saving embeddings"):
            embedding_blob = embeddings[i].tobytes()
            cursor.execute("""
                INSERT INTO centroids (prompt_id, embedding_vector, dimensions)
                VALUES (?, ?, ?)
            """, (prompt_id, embedding_blob, len(embeddings[i])))
        
        print(f"✅ Saved {len(embeddings)} embeddings to centroids table")

    # Update cluster_hierarchy table
    # First, clear existing hierarchy
    cursor.execute("DELETE FROM cluster_hierarchy")

    # Insert new hierarchy
    for i, fc in tqdm(enumerate(state['final_clusters']), desc="Saving cluster hierarchy"):
        cluster_id = fc['path']
        cluster_label = state['cluster_labels'][i]
        depth = len(cluster_id.split('/')) - 1 if '/' in cluster_id else 0
        prompt_count = len(fc['prompts'])

        # Parse path for levels
        path_parts = cluster_id.split('/')
        levels = {}
        for j in range(10):  # Up to level 9
            levels[f'level_{j}'] = path_parts[j] if j < len(path_parts) else None

        cursor.execute("""
            INSERT INTO cluster_hierarchy
            (cluster_id, cluster_label, depth, prompt_count,
             level_0, level_1, level_2, level_3, level_4,
             level_5, level_6, level_7, level_8, level_9)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cluster_id, cluster_label, depth, prompt_count,
            levels.get('level_0'), levels.get('level_1'), levels.get('level_2'),
            levels.get('level_3'), levels.get('level_4'), levels.get('level_5'),
            levels.get('level_6'), levels.get('level_7'), levels.get('level_8'),
            levels.get('level_9')
        ))

    conn.commit()
    conn.close()

    print(f"✅ Updated {updated_count} prompts, saved embeddings, and inserted {len(state['final_clusters'])} clusters in {db_path}")

    return state


def build_clustering_workflow_sqlite():
    """Build clustering workflow adapted for SQLite."""

    # Create the graph
    workflow = StateGraph(ClusteringState)

    # Add nodes (embed_prompts and later ones can stay the same)
    workflow.add_node("embed_prompts", embed_prompts)
    workflow.add_node("optimize_kmeans", optimize_kmeans)
    workflow.add_node("cluster_data", cluster_data)
    workflow.add_node("hierarchical_clustering", hierarchical_clustering)
    workflow.add_node("label_clusters", label_clusters)
    workflow.add_node("save_results_sqlite", save_results_sqlite)

    # Define the edges
    workflow.set_entry_point("embed_prompts")
    workflow.add_edge("embed_prompts", "optimize_kmeans")
    workflow.add_edge("optimize_kmeans", "cluster_data")
    workflow.add_edge("cluster_data", "hierarchical_clustering")
    workflow.add_edge("hierarchical_clustering", "label_clusters")
    workflow.add_edge("label_clusters", "save_results_sqlite")
    workflow.add_edge("save_results_sqlite", END)

    # Compile the graph
    app = workflow.compile()

    return app


def run_clustering_on_db(db_path: str, use_llm_labeling: bool = False, force_embeddings: bool = True):
    """Run clustering workflow on a SQLite database."""
    print("=" * 80)
    print(f"Clustering Workflow for {db_path}")
    print("=" * 80)

    # Load data
    initial_state = load_data_sqlite(db_path)
    initial_state['use_llm_labeling'] = use_llm_labeling

    # Force embeddings by deleting cached file
    if force_embeddings and os.path.exists(initial_state['embeddings_file']):
        print(f"Deleting cached embeddings: {initial_state['embeddings_file']}")
        os.remove(initial_state['embeddings_file'])

    # Build and run workflow
    app = build_clustering_workflow_sqlite()
    final_state = app.invoke(initial_state)

    print("=" * 80)
    print("Workflow completed successfully!")
    print(f"Processed {len(final_state['prompts'])} prompts")
    print(f"Created {len(final_state['final_clusters'])} final clusters")
    print("=" * 80)

    return final_state


if __name__ == "__main__":
    # Run clustering on benign database only
    run_clustering_on_db('naamse_benign.db', use_llm_labeling=False, force_embeddings=True)