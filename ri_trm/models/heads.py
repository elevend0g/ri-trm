"""
Output and Confidence Heads for RI-TRM

Converts latent representations back to discrete tokens and confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OutputHead(nn.Module):
    """
    Output head that converts latent representations to token logits
    
    Maps from continuous latent space back to discrete vocabulary for
    generating solution tokens.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        share_embedding_weights: bool = True,
        input_embedding: nn.Embedding = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Pre-projection normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        if share_embedding_weights and input_embedding is not None:
            # Share weights with input embedding (common practice)
            self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
            self.output_projection.weight = input_embedding.weight
        else:
            self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
            self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.normal_(self.output_projection.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to output logits
        
        Args:
            hidden_states: Latent representations [B, L, D]
            
        Returns:
            Token logits [B, L, V]
        """
        # Normalize and project
        normalized = self.norm(hidden_states)
        logits = self.output_projection(normalized)
        
        return logits
    
    def generate_tokens(
        self, 
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """
        Generate discrete tokens from hidden states
        
        Args:
            hidden_states: Latent representations [B, L, D]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated token IDs [B, L]
        """
        logits = self.forward(hidden_states)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample tokens
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
        
        return tokens
    
    def get_argmax_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get most likely tokens (greedy decoding)
        
        Args:
            hidden_states: Latent representations [B, L, D]
            
        Returns:
            Most likely token IDs [B, L]
        """
        logits = self.forward(hidden_states)
        return torch.argmax(logits, dim=-1)


class ConfidenceHead(nn.Module):
    """
    Confidence head that predicts model confidence in current solution
    
    Used for early stopping in recursive refinement and adaptive computation.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Confidence prediction network
        self.confidence_net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize confidence network weights"""
        for module in self.confidence_net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence score for current solution
        
        Args:
            hidden_states: Latent representations [B, L, D]
            
        Returns:
            Confidence scores [B, L, 1]
        """
        return self.confidence_net(hidden_states)
    
    def get_sequence_confidence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get overall confidence for the entire sequence
        
        Args:
            hidden_states: Latent representations [B, L, D]
            
        Returns:
            Sequence confidence scores [B, 1]
        """
        token_confidences = self.forward(hidden_states)  # [B, L, 1]
        
        # Take mean confidence across sequence
        sequence_confidence = token_confidences.mean(dim=1)  # [B, 1]
        
        return sequence_confidence
    
    def should_stop(
        self, 
        hidden_states: torch.Tensor, 
        threshold: float = 0.8,
        min_confidence: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine if recursive refinement should stop
        
        Args:
            hidden_states: Current latent state [B, L, D]
            threshold: Confidence threshold for stopping
            min_confidence: Minimum acceptable confidence
            
        Returns:
            (should_stop, confidence): Stop decisions [B] and confidence scores [B]
        """
        sequence_confidence = self.get_sequence_confidence(hidden_states).squeeze(-1)  # [B]
        
        # Stop if confidence is above threshold
        should_stop = sequence_confidence > threshold
        
        # But don't stop if confidence is too low (indicates uncertainty)
        low_confidence = sequence_confidence < min_confidence
        should_stop = should_stop & ~low_confidence
        
        return should_stop, sequence_confidence


class MultiTaskHead(nn.Module):
    """
    Multi-task head that combines output generation and confidence prediction
    
    Efficiently shares computation between output and confidence prediction.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        share_embedding_weights: bool = True,
        input_embedding: nn.Embedding = None
    ):
        super().__init__()
        
        # Shared feature extraction
        self.shared_norm = nn.LayerNorm(hidden_size)
        self.shared_projection = nn.Linear(hidden_size, hidden_size)
        
        # Task-specific heads
        self.output_head = OutputHead(
            hidden_size, vocab_size, share_embedding_weights, input_embedding
        )
        self.confidence_head = ConfidenceHead(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both outputs and confidence scores
        
        Args:
            hidden_states: Latent representations [B, L, D]
            
        Returns:
            (output_logits, confidence_scores): Output logits [B, L, V] and confidence [B, L, 1]
        """
        # Shared processing
        shared_features = self.shared_norm(hidden_states)
        shared_features = F.gelu(self.shared_projection(shared_features))
        
        # Task-specific outputs
        output_logits = self.output_head(shared_features)
        confidence_scores = self.confidence_head(shared_features)
        
        return output_logits, confidence_scores