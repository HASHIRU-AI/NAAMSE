import json
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import optuna
from typing import TypedDict, List, Dict, Any, Optional
from .types import ClusteringState
# Node 3: Optimize K-means parameters
def optimize_kmeans(state: ClusteringState) -> ClusteringState:
    """Optimize K-means parameters using Optuna."""
    params_file = f"{state['checkpoint_dir']}/kmeans_params.json"

    if os.path.exists(params_file):
        print("Loading cached K-means parameters...")
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded params: {best_params}")
    else:
        embeddings = state['embeddings']
        sample_size = min(5000, len(state['prompts']))
        sample_indices = np.random.choice(len(state['prompts']), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        print(f"Optimizing K-means on sample of {sample_size} prompts")

        def objective(trial):
            n_clusters = trial.suggest_int('n_clusters', 5, 30)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_embeddings)

            # Use silhouette score (higher is better)
            if len(set(labels)) > 1:
                score = silhouette_score(sample_embeddings, labels)
                return score  # Optuna maximizes by default
            else:
                return -1.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        print(f"Best params: {best_params}")

        # Save params for future runs
        with open(params_file, 'w') as f:
            json.dump(best_params, f)
        print(f"Saved params to {params_file}")

    return {
        **state,
        'best_params': best_params
    }