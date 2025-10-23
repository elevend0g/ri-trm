"""
Tiny Recursive Network (7M parameters) - Core of RI-TRM

Based on Samsung's TRM architecture with 2-layer Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.max_seq_len = max_seq_len
        self._init_cache(max_seq_len)
    
    def _init_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._init_cache(seq_len * 2)
        
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def apply_rotary_embedding(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim * 4, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 4, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary embeddings"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, L)
        q, k = apply_rotary_embedding(q, k, cos, sin)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attention(self.attention_norm(x))
        
        # MLP with residual connection  
        x = x + self.mlp(self.mlp_norm(x))
        
        return x


class TinyRecursiveNetwork(nn.Module):
    """
    Tiny Recursive Network - Core of RI-TRM
    
    A 2-layer transformer with ~7M parameters that performs recursive reasoning
    by iteratively improving solutions through latent space reasoning.
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        vocab_size: int = 32000
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size)
    
    def forward(
        self, 
        x: torch.Tensor,           # Input question embedding
        y: torch.Tensor,           # Current solution embedding  
        z: torch.Tensor,           # Latent reasoning state
        violations: Optional[torch.Tensor] = None,  # Rule violations
        candidate_paths: Optional[torch.Tensor] = None  # Path memory suggestions
    ) -> torch.Tensor:
        """
        Single reasoning step: improve latent state z given context
        
        Args:
            x: Input question embedding [B, L, D]
            y: Current solution embedding [B, L, D] 
            z: Latent reasoning state [B, L, D]
            violations: Rule violation embeddings [B, V, D]
            candidate_paths: Path memory suggestions [B, P, D]
            
        Returns:
            Updated latent state z [B, L, D]
        """
        # Combine inputs for reasoning context
        # Note: This follows TRM's approach of including x, y, z in reasoning
        reasoning_input = z + x + y  # Simple addition as in TRM
        
        # Add violation and path information if available
        if violations is not None:
            # Average violation embeddings and add to reasoning input
            violation_avg = violations.mean(dim=1, keepdim=True)  # [B, 1, D]
            reasoning_input = reasoning_input + violation_avg
            
        if candidate_paths is not None:
            # Weighted average of candidate paths (weights should come from path memory)
            path_avg = candidate_paths.mean(dim=1, keepdim=True)  # [B, 1, D]
            reasoning_input = reasoning_input + path_avg
        
        # Pass through transformer layers
        hidden = reasoning_input
        for layer in self.layers:
            hidden = layer(hidden)
        
        # Apply final normalization
        hidden = self.norm(hidden)
        
        return hidden
    
    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor, 
        z: torch.Tensor,
        n: int = 6,
        violations: Optional[torch.Tensor] = None,
        candidate_paths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform n steps of latent reasoning followed by solution update
        
        This implements the core recursion from TRM:
        1. Update latent state z for n iterations
        2. Update solution y based on final latent state
        
        Args:
            x: Input question embedding
            y: Current solution embedding
            z: Current latent state
            n: Number of reasoning steps
            violations: Current rule violations
            candidate_paths: Suggested paths from memory
            
        Returns:
            (updated_y, updated_z): New solution and latent state
        """
        # Step 1: Perform n latent reasoning updates
        current_z = z
        for i in range(n):
            current_z = self.forward(x, y, current_z, violations, candidate_paths)
        
        # Step 2: Update solution y based on current latent state
        # In TRM, this is done by f_O(z), but we integrate it here
        # The network learns to output y when given (y + z) without x
        solution_input = y + current_z  # No x indicates solution update mode
        updated_y = solution_input
        for layer in self.layers:
            updated_y = layer(updated_y)
        updated_y = self.norm(updated_y)
        
        return updated_y, current_z
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)