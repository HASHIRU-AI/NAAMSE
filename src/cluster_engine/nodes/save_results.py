import json
import os
import torch
import numpy as np
import hashlib
import pickle
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 7: Save results to files
def save_results(state: ClusteringState) -> ClusteringState:
    """Update main corpus with cluster metadata and save centroids."""
    print("Updating main corpus with cluster metadata...")

    # Create mapping from prompt to cluster info
    prompt_to_cluster = {}
    for i, fc in enumerate(state['final_clusters']):
        cluster_id = fc['path']
        cluster_label = state['cluster_labels'][i]
        centroid_coord = fc['centroid']
        for prompt in fc['prompts']:
            prompt_to_cluster[prompt] = {
                'cluster_id': cluster_id,
                'cluster_label': cluster_label,
                'centroid_coord': centroid_coord
            }

    # Read original corpus and update with cluster metadata
    updated_corpus = []
    with open('jailbreak_corpus.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data['messages'][0]['content']
            if prompt in prompt_to_cluster:
                cluster_info = prompt_to_cluster[prompt]
                data.update(cluster_info)
            updated_corpus.append(data)

    # Write back to the main corpus file
    with open('jailbreak_corpus.jsonl', 'w') as f:
        for data in updated_corpus:
            json.dump(data, f)
            f.write('\n')

    print(f"✅ Updated {len(updated_corpus)} entries in jailbreak_corpus.jsonl")

    # Optionally create organized cluster view files for browsing
    os.makedirs('clusters', exist_ok=True)
    for i, fc in enumerate(state['final_clusters']):
        label_name = state['cluster_labels'][i]
        centroid_hash = hashlib.md5(str(fc['centroid']).encode()).hexdigest()[:8]
        cluster_dir = f'clusters/{fc["path"]}'
        os.makedirs(cluster_dir, exist_ok=True)
        filename = f'{cluster_dir}/{label_name.replace(" ", "_")}_{centroid_hash}.jsonl'
        with open(filename, 'w', encoding='utf-8') as f:
            for p, s in zip(fc['prompts'], fc['sources']):
                json.dump({"source": s, "messages": [{"role": "user", "content": p}]}, f)
                f.write('\n')
        print(f"Created cluster view file: {filename}")

    # Save centroids
    centroids = {i: np.array(fc['centroid']) for i, fc in enumerate(state['final_clusters'])}
    with open('centroids.pkl', 'wb') as f:
        pickle.dump(centroids, f)
    print("Centroids saved to centroids.pkl")

    print("✅ All results saved successfully!")

    return state