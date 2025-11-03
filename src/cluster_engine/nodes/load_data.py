import json
import os
import torch
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 1: Load data from corpus
def load_data(state: ClusteringState) -> ClusteringState:
    """Load prompts and sources from the jailbreak corpus."""
    print("Loading corpus...")
    prompts = []
    sources = []

    with open('jailbreak_corpus.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['messages'][0]['content'])
            sources.append(data['source'])

    print(f"Loaded {len(prompts)} prompts")

    return {
        **state,
        'prompts': prompts,
        'sources': sources,
        'embeddings_file': 'embeddings.npy',
        'checkpoint_dir': 'checkpoints',
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'use_llm_labeling': state.get('use_llm_labeling', False)
    }