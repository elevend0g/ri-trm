"""
Path Selector for RI-TRM

Implements path selection strategies for the recursive refinement process.
This was referenced in the recursive_solver but wasn't created yet.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np

# Import the ReasoningPath from path_memory
from ..knowledge.path_memory import ReasoningPath


class PathSelector(ABC):
    """Abstract interface for path selection strategies"""
    
    @abstractmethod
    def select_path(self, candidate_paths: List[ReasoningPath], epsilon: float) -> Optional[ReasoningPath]:
        """Select a path from candidates using the selection strategy"""
        pass


class EpsilonGreedySelector(PathSelector):
    """ε-greedy path selection strategy"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def select_path(self, candidate_paths: List[ReasoningPath], epsilon: float) -> Optional[ReasoningPath]:
        """Select path using ε-greedy strategy"""
        if not candidate_paths:
            return None
        
        if np.random.random() < epsilon:
            # Exploration: random selection
            return np.random.choice(candidate_paths)
        else:
            # Exploitation: select highest weight
            return max(candidate_paths, key=lambda p: p.weight)


class UCBSelector(PathSelector):
    """Upper Confidence Bound path selection"""
    
    def __init__(self, c: float = 1.0):
        self.c = c  # Exploration parameter
    
    def select_path(self, candidate_paths: List[ReasoningPath], epsilon: float) -> Optional[ReasoningPath]:
        """Select path using UCB1 algorithm"""
        if not candidate_paths:
            return None
        
        total_usage = sum(p.usage_count for p in candidate_paths)
        if total_usage == 0:
            return np.random.choice(candidate_paths)
        
        def ucb_score(path: ReasoningPath) -> float:
            if path.usage_count == 0:
                return float('inf')
            
            confidence = self.c * np.sqrt(np.log(total_usage) / path.usage_count)
            return path.weight + confidence
        
        return max(candidate_paths, key=ucb_score)