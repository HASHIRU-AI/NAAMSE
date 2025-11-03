import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 2: Embed prompts
def embed_prompts(state: ClusteringState) -> ClusteringState:
    """Generate or load embeddings for prompts."""
    embeddings_file = state['embeddings_file']

    if os.path.exists(embeddings_file):
        print("Loading cached embeddings...")
        embeddings = np.load(embeddings_file)
    else:
        print(f"Using device: {state['device']}")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=state['device'])
        embeddings = model.encode(state['prompts'], show_progress_bar=True)
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved to {embeddings_file}")

    print("Embeddings ready")

    return {
        **state,
        'embeddings': embeddings
    }