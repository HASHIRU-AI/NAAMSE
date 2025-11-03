import json
import os
import torch
import numpy as np
import requests
import time
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 6: Label clusters with LLM
def label_clusters(state: ClusteringState) -> ClusteringState:
    """Label each cluster using LLM."""

    # Skip if LLM labeling is disabled
    if not state.get('use_llm_labeling', False):
        print("LLM labeling disabled, using generic labels...")
        cluster_labels = {i: f"Cluster_{i}" for i in range(len(state['final_clusters']))}
        return {
            **state,
            'cluster_labels': cluster_labels
        }

    def label_cluster(cluster_prompts, already_assigned_labels, max_retries=3):
        examples = cluster_prompts[:5]
        base_prompt = f"Here are examples of prompts in a cluster:\n" + "\n".join(examples) + "\n\nProvide a JSON object with a 'label' field containing a unique, yet descriptive phrase for how the prompts achieve bad behavior in this cluster. An example is \"Role-Play\"."

        for attempt in range(max_retries):
            try:
                # Add error message if this is a retry due to duplicate
                if attempt > 0:
                    prompt = base_prompt + f"\n\nERROR: The label you provided was already used. These labels are already taken: {', '.join(already_assigned_labels)}. Please provide a DIFFERENT, unique label."
                else:
                    prompt = base_prompt

                response = requests.post('http://localhost:11434/api/chat', json={
                    'model': 'llama3.2:3b',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'format': {
                        'type': 'object',
                        'properties': {
                            'label': {'type': 'string'}
                        },
                        'required': ['label']
                    },
                    'stream': False,
                    'options': {'temperature': 0.7}
                }, timeout=30)
                content = response.json()['message']['content']
                parsed = json.loads(content)
                label = parsed['label'].strip()

                # Check if label is already used
                if label in already_assigned_labels:
                    print(f"  Warning: LLM suggested duplicate label '{label}', retrying...")
                    continue

                return label

            except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException) as e:
                print(f"  Warning: LLM labeling failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        # Fallback to generic label
        return f"Cluster_Technique_{len(already_assigned_labels)}"

    # Load existing labels if available
    labels_file = f"{state['checkpoint_dir']}/cluster_labels.json"
    if os.path.exists(labels_file):
        print("Loading existing cluster labels...")
        with open(labels_file, 'r') as f:
            cluster_labels = {int(k): v for k, v in json.load(f).items()}
        start_idx = len(cluster_labels)
        print(f"Resuming from cluster {start_idx}")
    else:
        cluster_labels = {}
        start_idx = 0

    print(f"Labeling clusters {start_idx} to {len(state['final_clusters'])-1}...")
    for i in range(start_idx, len(state['final_clusters'])):
        fc = state['final_clusters'][i]
        cluster_labels[i] = label_cluster(fc['prompts'], list(cluster_labels.values()))
        print(f"Cluster {i}/{len(state['final_clusters'])-1}: {len(fc['prompts'])} prompts - {cluster_labels[i]}")

        # Save progress after each label
        with open(labels_file, 'w') as f:
            json.dump(cluster_labels, f, indent=2)

    return {
        **state,
        'cluster_labels': cluster_labels
    }