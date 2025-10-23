"""
Input and Output Embeddings for RI-TRM

Handles conversion between token sequences and continuous representations.
"""

import torch
import torch.nn as nn
from typing import Optional


class InputEmbedding(nn.Module):
    """
    Input embedding layer that converts token sequences to continuous representations
    
    Maps task specifications from discrete tokens to dense vectors suitable
    for the recursive reasoning network.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int = 2048,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, 
            hidden_size, 
            padding_idx=padding_idx
        )
        
        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert input tokens to embeddings
        
        Args:
            input_ids: Token IDs [B, L]
            
        Returns:
            Embedded input [B, L, D]
        """
        B, L = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine and normalize
        embeddings = token_embeds + position_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class OutputEmbedding(nn.Module):
    """
    Output embedding layer for solutions
    
    Similar to input embedding but specialized for solution representations.
    May share weights with input embedding for parameter efficiency.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int = 2048,
        padding_idx: Optional[int] = None,
        share_input_embedding: Optional[nn.Embedding] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Token embeddings (optionally shared)
        if share_input_embedding is not None:
            self.token_embedding = share_input_embedding
        else:
            self.token_embedding = nn.Embedding(
                vocab_size, 
                hidden_size, 
                padding_idx=padding_idx
            )
        
        # Separate position embeddings for solutions
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Solution-specific normalization
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        if share_input_embedding is None:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, solution_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert solution tokens to embeddings
        
        Args:
            solution_ids: Solution token IDs [B, L]
            
        Returns:
            Embedded solution [B, L, D]
        """
        B, L = solution_ids.shape
        
        # Create position indices
        position_ids = torch.arange(L, device=solution_ids.device).unsqueeze(0).expand(B, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(solution_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine and normalize
        embeddings = token_embeds + position_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def init_random_solution(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Initialize random solution embeddings for starting the recursive process
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            Random solution embeddings [B, L, D]
        """
        # Create random token IDs (avoiding padding and special tokens)
        random_ids = torch.randint(
            1, self.vocab_size - 1, 
            (batch_size, seq_len), 
            device=device
        )
        
        return self.forward(random_ids)


class LatentEmbedding(nn.Module):
    """
    Latent reasoning state embedding
    
    Initializes and manages the latent reasoning state z that evolves
    during recursive reasoning.
    """
    
    def __init__(self, hidden_size: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Learnable initialization for latent state
        self.init_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # Position embeddings for latent state
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Initialize latent reasoning state
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length  
            device: Device to create tensors on
            
        Returns:
            Initial latent state [B, L, D]
        """
        # Expand initialization embedding
        init_z = self.init_embedding.expand(batch_size, seq_len, -1)
        
        # Add position embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine and normalize
        latent_state = init_z + position_embeds
        latent_state = self.norm(latent_state)
        
        return latent_state