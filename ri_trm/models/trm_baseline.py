"""
TRM Baseline Model (Without Rules) - For Ablation Studies

This is a baseline implementation of the Tiny Recursive Model WITHOUT
the explicit rule verification (K_R) and path memory (K_P) components.

Purpose: Measure the contribution of RI-TRM's key innovations:
- Explicit rule verification (K_R)
- Hebbian path memory (K_P)

By comparing RI-TRM vs TRM Baseline, we can quantify the gains from
structured knowledge vs pure learning.

Based on instructions.md Priority 3: TRM Baseline
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

from .network import TinyRecursiveNetwork
from .embedding import InputEmbedding, OutputEmbedding, LatentEmbedding
from .heads import OutputHead, ConfidenceHead
from ..inference.recursive_solver import ReasoningStep, RefinementResult


class TRMBaseline(nn.Module):
    """
    Tiny Recursive Model Baseline (No Explicit Rules)

    Key differences from RI-TRM:
    - NO rule verification (K_R disabled)
    - NO path memory guidance (K_P disabled)
    - Pure end-to-end learning
    - Same architecture (for fair comparison)

    This allows measuring: Accuracy_RI-TRM - Accuracy_TRM = Gain from K_R + K_P
    """

    def __init__(
        self,
        network: TinyRecursiveNetwork,
        input_embedding: InputEmbedding,
        output_embedding: OutputEmbedding,
        latent_embedding: LatentEmbedding,
        output_head: OutputHead,
        confidence_head: ConfidenceHead,
        vocab_size: int,
        max_iterations: int = 16,
        reasoning_steps: int = 6,
        confidence_threshold: float = 0.8,
        min_confidence: float = 0.3,
        use_rules: bool = False  # KEY PARAMETER: Always False for baseline
    ):
        super().__init__()

        # Core components (identical architecture to RI-TRM)
        self.network = network
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.latent_embedding = latent_embedding
        self.output_head = output_head
        self.confidence_head = confidence_head

        # Hyperparameters
        self.vocab_size = vocab_size
        self.max_iterations = max_iterations
        self.reasoning_steps = reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence

        # CRITICAL: No rule verification or path memory
        self.use_rules = use_rules  # Always False for baseline
        assert not use_rules, "TRM Baseline should not use rules (use_rules=False)"

        # These remain None (no knowledge components)
        self.rule_verifier = None
        self.path_memory = None
        self.factual_kg = None

    def initial_draft(
        self,
        task_embedding: torch.Tensor,
        task_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate initial solution WITHOUT rule guidance

        Unlike RI-TRM, this uses only the neural network's learned patterns,
        not explicit rule templates.

        Args:
            task_embedding: Embedded task specification [B, L, D]
            task_tokens: Original task tokens [B, L]

        Returns:
            Initial solution embedding [B, L, D]
        """
        batch_size, seq_len, hidden_size = task_embedding.shape

        # Pure neural initialization (no rule templates)
        # Just pass through the network once to get initial guess
        initial_solution = task_embedding.clone()  # Start from task

        # Single forward pass for initial draft
        with torch.no_grad():
            for step in range(2):  # Quick 2-step initialization
                # Process through network
                task_attended, solution_attended = self.network(
                    task_embedding,
                    initial_solution,
                    initial_solution
                )

                # Update solution
                initial_solution = solution_attended

        return initial_solution

    def verify_and_refine(
        self,
        task_embedding: torch.Tensor,
        solution_embedding: torch.Tensor,
        latent_state: torch.Tensor,
        task_tokens: torch.Tensor,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ReasoningStep]:
        """
        Refine solution WITHOUT rule verification

        This is pure neural refinement - no explicit rule checking,
        no path memory guidance. The network learns to self-correct
        purely from training data.

        Args:
            task_embedding: Task embedding [B, L, D]
            solution_embedding: Current solution embedding [B, L, D]
            latent_state: Latent reasoning state [B, L, D]
            task_tokens: Task tokens [B, L]
            step: Current refinement step

        Returns:
            Tuple of (refined_solution, updated_latent, reasoning_step)
        """
        # NO RULE VERIFICATION - key difference from RI-TRM
        violations = []  # Always empty for baseline

        # NO PATH MEMORY GUIDANCE - another key difference
        selected_path = None
        path_weight = 0.0

        # Pure neural refinement
        # Process through recursive network
        task_attended, solution_attended = self.network(
            task_embedding,
            solution_embedding,
            latent_state
        )

        # Update latent state (pure learning)
        updated_latent = latent_state + solution_attended

        # Refine solution (no rule-guided corrections)
        refined_solution = solution_embedding + solution_attended * 0.1

        # Compute confidence (no rule-based adjustments)
        confidence = self.confidence_head(refined_solution.mean(dim=1))

        # Record reasoning step
        reasoning_step = ReasoningStep(
            step=step,
            violations=[],  # No rule checking
            selected_path=None,  # No path memory
            path_weight=0.0,
            confidence=confidence.mean().item(),
            transformation="pure_neural_refinement",
            success=confidence.mean().item() > self.confidence_threshold,
            latent_state=updated_latent.detach(),
            solution_state=refined_solution.detach()
        )

        return refined_solution, updated_latent, reasoning_step

    def forward(
        self,
        task_tokens: torch.Tensor,
        return_trace: bool = False,
        early_stopping: bool = True
    ) -> RefinementResult:
        """
        Recursive refinement WITHOUT rules

        Args:
            task_tokens: Input task tokens [B, L]
            return_trace: Whether to return full reasoning trace
            early_stopping: Stop when confidence threshold reached

        Returns:
            RefinementResult with solution and trace
        """
        start_time = time.time()

        # Embed task
        task_embedding = self.input_embedding(task_tokens)

        # Initialize solution (no rule guidance)
        solution_embedding = self.initial_draft(task_embedding, task_tokens)

        # Initialize latent state
        latent_state = self.latent_embedding(torch.zeros_like(task_tokens))

        # Recursive refinement loop (NO RULES)
        reasoning_trace = []
        converged = False

        for step in range(self.max_iterations):
            # Refine without rule verification
            solution_embedding, latent_state, reasoning_step = self.verify_and_refine(
                task_embedding,
                solution_embedding,
                latent_state,
                task_tokens,
                step
            )

            if return_trace:
                reasoning_trace.append(reasoning_step)

            # Early stopping based on pure confidence
            if early_stopping and reasoning_step.confidence > self.confidence_threshold:
                converged = True
                break

        # Generate final solution tokens
        solution_logits = self.output_head(solution_embedding)
        solution_tokens = torch.argmax(solution_logits, dim=-1)

        # Final confidence
        final_confidence = self.confidence_head(solution_embedding.mean(dim=1)).mean().item()

        total_time = time.time() - start_time

        return RefinementResult(
            solution=solution_tokens,
            reasoning_trace=reasoning_trace if return_trace else [],
            final_confidence=final_confidence,
            num_steps=len(reasoning_trace),
            converged=converged,
            total_time=total_time
        )

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_knowledge_components(self, rule_verifier, path_memory, factual_kg=None):
        """
        Dummy method for API compatibility

        TRM Baseline DOES NOT use knowledge components, but we keep
        this method for API compatibility with RI-TRM.
        """
        # Intentionally do nothing - baseline doesn't use knowledge components
        pass


def create_trm_baseline(
    vocab_size: int,
    hidden_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 8,
    max_seq_len: int = 512,
    max_iterations: int = 16,
    reasoning_steps: int = 6,
    device: str = "cpu"
) -> TRMBaseline:
    """
    Factory function to create TRM Baseline model

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of network layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        max_iterations: Maximum refinement iterations
        reasoning_steps: Number of reasoning steps
        device: Device to place model on

    Returns:
        TRMBaseline model (without rules)
    """
    # Create network components (identical to RI-TRM)
    network = TinyRecursiveNetwork(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size
    ).to(device)

    input_embedding = InputEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len
    ).to(device)

    output_embedding = OutputEmbedding(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        share_input_embedding=input_embedding.token_embedding
    ).to(device)

    latent_embedding = LatentEmbedding(
        hidden_size=hidden_size,
        max_seq_len=max_seq_len
    ).to(device)

    output_head = OutputHead(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        share_embedding_weights=True,
        input_embedding=input_embedding.token_embedding
    ).to(device)

    confidence_head = ConfidenceHead(
        hidden_size=hidden_size
    ).to(device)

    # Create baseline (use_rules=False)
    baseline = TRMBaseline(
        network=network,
        input_embedding=input_embedding,
        output_embedding=output_embedding,
        latent_embedding=latent_embedding,
        output_head=output_head,
        confidence_head=confidence_head,
        vocab_size=vocab_size,
        max_iterations=max_iterations,
        reasoning_steps=reasoning_steps,
        use_rules=False  # Critical: no rules for baseline
    ).to(device)

    return baseline
