"""
Global configuration and seeding utilities for reproducible runs.

This module provides centralized configuration management and random seed
setting across all randomness sources used in the project.
"""
import os
import random
import numpy as np
import torch
from typing import Optional

# Load .env file before reading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, rely on system environment variables
    pass


class Config:
    """Global configuration for the fuzzing framework."""
    
    # Default seed value (None means non-deterministic)
    DEFAULT_SEED: Optional[int] = None
    
    # Current active seed
    _seed: Optional[int] = None
    
    @classmethod
    def set_seed(cls, seed: Optional[int] = None) -> None:
        """
        Set random seed for all randomness sources for reproducible runs.
        
        This sets seeds for:
        - Python's random module
        - NumPy's random number generator
        - PyTorch (CPU and CUDA)
        
        Args:
            seed: Random seed to use. If None, uses non-deterministic behavior.
        
        Example:
            >>> from src.config import Config
            >>> Config.set_seed(42)  # All random operations are now reproducible
        """
        cls._seed = seed
        
        if seed is not None:
            # Python random
            random.seed(seed)
            
            # NumPy
            np.random.seed(seed)
            
            # PyTorch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Make PyTorch deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            print(f"🌱 Random seed set to {seed} for reproducible runs")
        else:
            print("🎲 Running in non-deterministic mode (no seed set)")
    
    @classmethod
    def get_seed(cls) -> Optional[int]:
        """
        Get the current random seed.
        
        Returns:
            The current seed value, or None if not set
        """
        return cls._seed
    
    @classmethod
    def initialize_from_env(cls) -> None:
        """
        Initialize configuration from environment variables.
        
        Looks for NAAMSE_RANDOM_SEED environment variable to set the seed.
        
        Example:
            $ export NAAMSE_RANDOM_SEED=42
            $ python -m src.agent.graph
        """
        seed_str = os.getenv("NAAMSE_RANDOM_SEED")
        if seed_str is not None:
            try:
                seed = int(seed_str)
                cls.set_seed(seed)
            except ValueError:
                print(f"⚠️  Warning: Invalid RANDOM_SEED value '{seed_str}', ignoring")
        elif cls.DEFAULT_SEED is not None:
            cls.set_seed(cls.DEFAULT_SEED)


# Auto-initialize from environment on import
# Config.initialize_from_env()
