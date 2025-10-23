"""
Layer 3: Path Memory Graph (K_P)

Hebbian-style path strengthening for learned debugging patterns.
Stores successful reasoning paths and strengthens them over time.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class ReasoningPath:
    """A single reasoning path from error state to solution"""
    id: str
    error_state: str  # Description of the error state
    action: str  # The transformation/fix applied
    result_state: str  # Description of resulting state
    weight: float = 0.5  # Success rate (0-1)
    usage_count: int = 0
    last_used: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathUpdate:
    """Record of a path weight update"""
    path_id: str
    old_weight: float
    new_weight: float
    success: bool
    update_type: str  # "ltp", "ltd", "myelination"


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


class PathMemoryGraph(nn.Module):
    """
    Layer 3: Path Memory Graph (K_P)
    
    Implements Hebbian-style path strengthening where successful reasoning
    paths are strengthened over time, creating interpretable "thick pathways"
    that explain model decisions.
    
    Path format: (error_state, action, result_state, weight)
    
    Examples for Code Generation:
    (IndentationError, Add_Indentation(4_spaces), Syntax_Valid, 0.96)
    (NameError, Import_Module(missing_module), Code_Runs, 0.89)
    (TypeError, Add_Type_Cast(str_to_int), Tests_Pass, 0.92)
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        max_paths: int = 50000,
        learning_rate: float = 0.1,
        myelination_boost: float = 1.1,
        decay_rate: float = 0.95,
        myelination_threshold: int = 10,
        epsilon_init: float = 0.3,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        selector_type: str = "epsilon_greedy"
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_paths = max_paths
        
        # Hebbian learning parameters
        self.learning_rate = learning_rate  # α
        self.myelination_boost = myelination_boost  # β
        self.decay_rate = decay_rate  # γ
        self.myelination_threshold = myelination_threshold  # θ
        
        # Exploration parameters
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Path storage
        self.paths: Dict[str, ReasoningPath] = {}
        self.error_to_paths: Dict[str, Set[str]] = defaultdict(set)
        self.update_history: List[PathUpdate] = []
        
        # Neural components for similarity matching
        self.state_embedding = nn.Embedding(10000, embedding_dim)  # Error states
        self.action_embedding = nn.Embedding(1000, embedding_dim)  # Actions
        
        # Embedding caches
        self.state_to_id: Dict[str, int] = {}
        self.action_to_id: Dict[str, int] = {}
        self.next_state_id = 0
        self.next_action_id = 0
        
        # Path selection strategy
        if selector_type == "epsilon_greedy":
            self.selector = EpsilonGreedySelector()
        elif selector_type == "ucb":
            self.selector = UCBSelector()
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")
    
    def _get_state_id(self, state: str) -> int:
        """Get or create embedding ID for error state"""
        if state not in self.state_to_id:
            self.state_to_id[state] = self.next_state_id % self.state_embedding.num_embeddings
            self.next_state_id += 1
        return self.state_to_id[state]
    
    def _get_action_id(self, action: str) -> int:
        """Get or create embedding ID for action"""
        if action not in self.action_to_id:
            self.action_to_id[action] = self.next_action_id % self.action_embedding.num_embeddings
            self.next_action_id += 1
        return self.action_to_id[action]
    
    def add_path(self, path: ReasoningPath):
        """Add a new reasoning path to memory"""
        if len(self.paths) >= self.max_paths:
            # Remove oldest/least successful path
            oldest_path = min(self.paths.values(), key=lambda p: (p.weight, p.last_used or 0))
            self.remove_path(oldest_path.id)
        
        self.paths[path.id] = path
        self.error_to_paths[path.error_state].add(path.id)
        
        # Cache embeddings
        self._get_state_id(path.error_state)
        self._get_action_id(path.action)
        self._get_state_id(path.result_state)
    
    def remove_path(self, path_id: str):
        """Remove a path from memory"""
        if path_id in self.paths:
            path = self.paths[path_id]
            self.error_to_paths[path.error_state].discard(path_id)
            del self.paths[path_id]
    
    def query_similar_states(
        self,
        error_states: List[str],
        current_latent: torch.Tensor,
        max_candidates: int = 10
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[str, float]]]:
        """
        Query path memory for states similar to current errors
        
        Args:
            error_states: Current error/violation descriptions
            current_latent: Current latent state for similarity matching
            max_candidates: Maximum number of candidate paths
            
        Returns:
            (path_embeddings, path_info): Embeddings and (action, weight) tuples
        """
        if not error_states:
            return None, []
        
        # Find candidate paths based on exact and similar error states
        candidate_paths = []
        
        for error_state in error_states:
            # Exact matches
            exact_paths = [self.paths[pid] for pid in self.error_to_paths.get(error_state, set())]
            candidate_paths.extend(exact_paths)
            
            # Similar states (simplified - could use embedding similarity)
            for state in self.error_to_paths.keys():
                if error_state in state or state in error_state:
                    similar_paths = [self.paths[pid] for pid in self.error_to_paths[state]]
                    candidate_paths.extend(similar_paths)
        
        # Remove duplicates and sort by weight
        unique_paths = {p.id: p for p in candidate_paths}
        sorted_paths = sorted(unique_paths.values(), key=lambda p: p.weight, reverse=True)
        top_paths = sorted_paths[:max_candidates]
        
        if not top_paths:
            return None, []
        
        # Create embeddings for candidate paths
        path_embeddings = self._embed_paths(top_paths)
        path_info = [(p.action, p.weight) for p in top_paths]
        
        return path_embeddings, path_info
    
    def _embed_paths(self, paths: List[ReasoningPath]) -> torch.Tensor:
        """Convert paths to neural embeddings"""
        embeddings = []
        
        for path in paths:
            # Get embeddings for error state and action
            state_id = self._get_state_id(path.error_state)
            action_id = self._get_action_id(path.action)
            
            state_emb = self.state_embedding(torch.tensor(state_id))
            action_emb = self.action_embedding(torch.tensor(action_id))
            
            # Combine state and action embeddings
            path_emb = state_emb + action_emb
            embeddings.append(path_emb)
        
        return torch.stack(embeddings)
    
    def select_path(self, candidate_paths: List[ReasoningPath]) -> Optional[ReasoningPath]:
        """Select a path using the configured selection strategy"""
        return self.selector.select_path(candidate_paths, self.epsilon)
    
    def update_path(
        self,
        error_states: List[str],
        selected_action: Optional[str],
        result_states: List[str],
        success: bool
    ):
        """
        Update path weights using Hebbian learning rules
        
        Args:
            error_states: Original error states
            selected_action: Action that was taken (None if no action)
            result_states: Resulting states after action
            success: Whether the action improved the situation
        """
        if not selected_action or not error_states:
            return
        
        # Find or create the path
        path_id = f"{error_states[0]}|{selected_action}|{result_states[0] if result_states else 'unknown'}"
        
        if path_id not in self.paths:
            # Create new path
            path = ReasoningPath(
                id=path_id,
                error_state=error_states[0],
                action=selected_action,
                result_state=result_states[0] if result_states else "unknown"
            )
            self.add_path(path)
        
        path = self.paths[path_id]
        old_weight = path.weight
        
        # Apply Hebbian learning rules
        if success:
            # Long-term potentiation (LTP)
            path.weight = path.weight + self.learning_rate * (1 - path.weight)
            path.usage_count += 1
            update_type = "ltp"
            
            # Myelination (strengthen heavily-used paths)
            if path.usage_count > self.myelination_threshold:
                path.weight = min(path.weight * self.myelination_boost, 0.99)
                update_type = "myelination"
        else:
            # Long-term depression (LTD)
            path.weight = path.weight * self.decay_rate
            update_type = "ltd"
        
        path.last_used = torch.randn(1).item()  # Timestamp placeholder
        
        # Record update for analysis
        update = PathUpdate(
            path_id=path_id,
            old_weight=old_weight,
            new_weight=path.weight,
            success=success,
            update_type=update_type
        )
        self.update_history.append(update)
        
        # Decay exploration parameter
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def get_path_statistics(self) -> Dict[str, Any]:
        """Get statistics about path memory"""
        if not self.paths:
            return {"total_paths": 0}
        
        weights = [p.weight for p in self.paths.values()]
        usage_counts = [p.usage_count for p in self.paths.values()]
        
        # Top paths by weight
        top_paths = sorted(self.paths.values(), key=lambda p: p.weight, reverse=True)[:10]
        top_path_info = [
            {
                "error_state": p.error_state,
                "action": p.action,
                "weight": p.weight,
                "usage_count": p.usage_count
            }
            for p in top_paths
        ]
        
        return {
            "total_paths": len(self.paths),
            "average_weight": np.mean(weights),
            "weight_std": np.std(weights),
            "max_weight": max(weights),
            "min_weight": min(weights),
            "average_usage": np.mean(usage_counts),
            "total_updates": len(self.update_history),
            "current_epsilon": self.epsilon,
            "top_paths": top_path_info,
            "unique_error_states": len(self.error_to_paths),
            "memory_usage": len(self.paths) / self.max_paths
        }
    
    def get_interpretable_trace(self, selected_paths: List[ReasoningPath]) -> List[Dict[str, Any]]:
        """
        Generate interpretable explanation of path selection
        
        Args:
            selected_paths: Paths that were selected during reasoning
            
        Returns:
            List of interpretable path descriptions
        """
        trace = []
        
        for i, path in enumerate(selected_paths):
            explanation = {
                "step": i + 1,
                "error_detected": path.error_state,
                "action_taken": path.action,
                "confidence": f"{path.weight:.2%}",
                "experience": f"Used {path.usage_count} times",
                "reasoning": f"This fix has {path.weight:.1%} success rate based on {path.usage_count} previous similar cases"
            }
            trace.append(explanation)
        
        return trace
    
    def save_memory(self, file_path: str):
        """Save path memory to file"""
        memory_data = {
            "paths": {pid: {
                "id": p.id,
                "error_state": p.error_state,
                "action": p.action,
                "result_state": p.result_state,
                "weight": p.weight,
                "usage_count": p.usage_count,
                "last_used": p.last_used,
                "metadata": p.metadata
            } for pid, p in self.paths.items()},
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "myelination_boost": self.myelination_boost,
                "decay_rate": self.decay_rate,
                "myelination_threshold": self.myelination_threshold,
                "epsilon": self.epsilon
            },
            "embeddings": {
                "state_to_id": self.state_to_id,
                "action_to_id": self.action_to_id
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, file_path: str):
        """Load path memory from file"""
        try:
            with open(file_path, 'r') as f:
                memory_data = json.load(f)
            
            # Load paths
            self.paths = {}
            self.error_to_paths = defaultdict(set)
            
            for pid, pdata in memory_data.get("paths", {}).items():
                path = ReasoningPath(**pdata)
                self.paths[pid] = path
                self.error_to_paths[path.error_state].add(pid)
            
            # Load hyperparameters
            hyperparams = memory_data.get("hyperparameters", {})
            for key, value in hyperparams.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Load embeddings
            embeddings = memory_data.get("embeddings", {})
            self.state_to_id = embeddings.get("state_to_id", {})
            self.action_to_id = embeddings.get("action_to_id", {})
            
        except Exception as e:
            print(f"Warning: Could not load path memory from {file_path}: {e}")
    
    def forward(self, error_states: List[str], current_latent: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Forward pass: query and embed relevant paths
        
        Args:
            error_states: Current error states
            current_latent: Current latent state
            
        Returns:
            Embedded candidate paths or None
        """
        path_embeddings, _ = self.query_similar_states(error_states, current_latent)
        return path_embeddings