"""
Recursive Refinement Solver - Core Algorithm 1 from RI-TRM

Implements the recursive refinement algorithm that combines TRM's deep supervision
with RI-TRM's explicit rule verification and path memory guidance.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

from ..models.network import TinyRecursiveNetwork
from ..models.embedding import InputEmbedding, OutputEmbedding, LatentEmbedding
from ..models.heads import OutputHead, ConfidenceHead


@dataclass
class ReasoningStep:
    """Record of a single reasoning step for interpretability"""
    step: int
    violations: List[str]
    selected_path: Optional[str]
    path_weight: float
    confidence: float
    transformation: str
    success: bool
    latent_state: torch.Tensor
    solution_state: torch.Tensor


@dataclass
class RefinementResult:
    """Result of recursive refinement process"""
    solution: torch.Tensor
    reasoning_trace: List[ReasoningStep]
    final_confidence: float
    num_steps: int
    converged: bool
    total_time: float


class RecursiveRefinementSolver(nn.Module):
    """
    RI-TRM Recursive Refinement Solver
    
    Implements Algorithm 1 from the RI-TRM paper:
    1. Initialize with rule-guided draft
    2. Iteratively verify and refine solution
    3. Use path memory for guidance
    4. Apply Hebbian strengthening
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
        min_confidence: float = 0.3
    ):
        super().__init__()
        
        # Core components
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
        
        # Knowledge components (to be injected)
        self.rule_verifier = None
        self.path_memory = None
        self.factual_kg = None
    
    def set_knowledge_components(self, rule_verifier, path_memory, factual_kg=None):
        """Inject knowledge components after initialization"""
        self.rule_verifier = rule_verifier
        self.path_memory = path_memory
        self.factual_kg = factual_kg
    
    def initial_draft(
        self,
        task_embedding: torch.Tensor,
        task_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate rule-guided initial solution draft

        Args:
            task_embedding: Embedded task specification [B, L, D]
            task_tokens: Original task tokens [B, L]

        Returns:
            Initial solution embedding [B, L, D]
        """
        batch_size, seq_len, hidden_size = task_embedding.shape

        if self.rule_verifier is not None:
            # Use rule knowledge to guide initial solution
            initial_tokens = self.rule_verifier.suggest_initial_solution(task_tokens)
            if initial_tokens is not None:
                # Pad or trim to match task sequence length
                current_len = initial_tokens.shape[1]
                if current_len < seq_len:
                    # Pad with zeros (padding token)
                    padding = torch.zeros(
                        (batch_size, seq_len - current_len),
                        dtype=initial_tokens.dtype,
                        device=initial_tokens.device
                    )
                    initial_tokens = torch.cat([initial_tokens, padding], dim=1)
                elif current_len > seq_len:
                    # Trim to match
                    initial_tokens = initial_tokens[:, :seq_len]

                return self.output_embedding(initial_tokens)

        # Fallback: random initialization
        return self.output_embedding.init_random_solution(
            batch_size, seq_len, task_embedding.device
        )
    
    def verify_solution(
        self, 
        solution_tokens: torch.Tensor,
        task_tokens: torch.Tensor
    ) -> Tuple[List[str], bool]:
        """
        Verify current solution against structural rules
        
        Args:
            solution_tokens: Current solution tokens [B, L]
            task_tokens: Original task tokens [B, L]
            
        Returns:
            (violations, is_valid): List of violation descriptions and validity flag
        """
        if self.rule_verifier is None:
            return [], True
            
        # Convert to text for rule verification
        violations = []
        for i in range(solution_tokens.shape[0]):
            batch_violations = self.rule_verifier.verify(
                solution_tokens[i].cpu().numpy(),
                task_tokens[i].cpu().numpy()
            )
            violations.extend(batch_violations)
        
        return violations, len(violations) == 0
    
    def query_path_memory(
        self,
        violations: List[str],
        current_state: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[str, float]]]:
        """
        Query path memory for similar violation patterns
        
        Args:
            violations: Current rule violations
            current_state: Current latent state [B, L, D]
            
        Returns:
            (candidate_paths, path_info): Path embeddings and path descriptions with weights
        """
        if self.path_memory is None or not violations:
            return None, []
        
        # Query path memory for similar states
        candidate_paths, path_info = self.path_memory.query_similar_states(
            violations, current_state
        )
        
        return candidate_paths, path_info
    
    def recursive_reasoning(
        self,
        task_embedding: torch.Tensor,
        solution_embedding: torch.Tensor,
        latent_state: torch.Tensor,
        violations: List[str],
        candidate_paths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform n steps of recursive reasoning
        
        Args:
            task_embedding: Task specification [B, L, D]
            solution_embedding: Current solution [B, L, D]
            latent_state: Current latent state [B, L, D]
            violations: Current violations
            candidate_paths: Suggested paths from memory [B, P, D]
            
        Returns:
            Updated latent state [B, L, D]
        """
        # Embed violations for network input
        violation_embedding = None
        if violations and self.rule_verifier:
            violation_embedding = self.rule_verifier.embed_violations(violations)
            if violation_embedding is not None:
                violation_embedding = violation_embedding.to(latent_state.device)
        
        # Perform recursive reasoning steps
        current_latent = latent_state
        for step in range(self.reasoning_steps):
            current_latent = self.network(
                task_embedding,
                solution_embedding, 
                current_latent,
                violation_embedding,
                candidate_paths
            )
        
        return current_latent
    
    def update_solution(
        self,
        current_solution: torch.Tensor,
        updated_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Update solution based on latent reasoning state
        
        Args:
            current_solution: Current solution embedding [B, L, D]
            updated_latent: Updated latent state [B, L, D]
            
        Returns:
            New solution embedding [B, L, D]
        """
        # Use network to generate new solution
        new_solution, _ = self.network.latent_recursion(
            torch.zeros_like(current_solution),  # No task input for solution update
            current_solution,
            updated_latent,
            n=1  # Single step for solution update
        )
        
        return new_solution
    
    def record_reasoning_step(
        self,
        step: int,
        violations: List[str],
        selected_path: Optional[str],
        path_weight: float,
        confidence: float,
        old_solution: torch.Tensor,
        new_solution: torch.Tensor,
        latent_state: torch.Tensor,
        success: bool
    ) -> ReasoningStep:
        """Record a reasoning step for interpretability"""
        
        # Describe transformation
        if torch.equal(old_solution, new_solution):
            transformation = "No change"
        else:
            transformation = "Solution modified"
        
        return ReasoningStep(
            step=step,
            violations=violations.copy(),
            selected_path=selected_path,
            path_weight=path_weight,
            confidence=confidence,
            transformation=transformation,
            success=success,
            latent_state=latent_state.clone().detach(),
            solution_state=new_solution.clone().detach()
        )
    
    def forward(
        self,
        task_tokens: torch.Tensor,
        return_trace: bool = True,
        early_stopping: bool = True
    ) -> RefinementResult:
        """
        Main recursive refinement algorithm (Algorithm 1)
        
        Args:
            task_tokens: Input task specification [B, L]
            return_trace: Whether to return detailed reasoning trace
            early_stopping: Whether to use confidence-based early stopping
            
        Returns:
            RefinementResult with solution and reasoning trace
        """
        start_time = time.time()
        
        batch_size, seq_len = task_tokens.shape
        device = task_tokens.device
        
        # Step 1: Embed task specification
        task_embedding = self.input_embedding(task_tokens)
        
        # Step 2: Initialize solution and latent state
        solution_embedding = self.initial_draft(task_embedding, task_tokens)
        latent_state = self.latent_embedding(batch_size, seq_len, device)
        
        # Initialize reasoning trace
        reasoning_trace = []
        
        # Step 3: Recursive refinement loop
        for step in range(self.max_iterations):
            
            # Convert current solution to tokens for verification
            solution_tokens = self.output_head.get_argmax_tokens(solution_embedding)
            
            # Step 4: Verify current solution
            violations, is_valid = self.verify_solution(solution_tokens, task_tokens)
            
            # Check for completion
            if is_valid:
                # Solution is valid, check if tests would pass (placeholder)
                confidence = self.confidence_head.get_sequence_confidence(latent_state).mean().item()
                
                if return_trace:
                    reasoning_trace.append(self.record_reasoning_step(
                        step, violations, None, 0.0, confidence,
                        solution_embedding, solution_embedding, latent_state, True
                    ))
                
                return RefinementResult(
                    solution=solution_tokens,
                    reasoning_trace=reasoning_trace,
                    final_confidence=confidence,
                    num_steps=step + 1,
                    converged=True,
                    total_time=time.time() - start_time
                )
            
            # Step 5: Query path memory for guidance
            candidate_paths, path_info = self.query_path_memory(violations, latent_state)
            
            # Select best path (ε-greedy handled by path memory)
            selected_path = path_info[0][0] if path_info else None
            path_weight = path_info[0][1] if path_info else 0.0
            
            # Step 6: Recursive reasoning
            old_solution = solution_embedding.clone()
            updated_latent = self.recursive_reasoning(
                task_embedding,
                solution_embedding,
                latent_state,
                violations,
                candidate_paths
            )
            
            # Step 7: Update solution
            new_solution = self.update_solution(solution_embedding, updated_latent)
            
            # Step 8: Record path and update memory
            new_solution_tokens = self.output_head.get_argmax_tokens(new_solution)
            new_violations, _ = self.verify_solution(new_solution_tokens, task_tokens)
            
            # Success if we reduced violations
            success = len(new_violations) < len(violations)
            
            if self.path_memory is not None:
                self.path_memory.update_path(
                    violations, selected_path, new_violations, success
                )
            
            # Update states
            solution_embedding = new_solution
            latent_state = updated_latent.detach()  # Detach for next iteration
            
            # Calculate confidence for early stopping
            confidence = self.confidence_head.get_sequence_confidence(latent_state).mean().item()
            
            # Record reasoning step
            if return_trace:
                reasoning_trace.append(self.record_reasoning_step(
                    step, violations, selected_path, path_weight, confidence,
                    old_solution, new_solution, latent_state, success
                ))
            
            # Early stopping based on confidence
            if early_stopping and confidence < self.min_confidence:
                break
        
        # Return final result
        final_tokens = self.output_head.get_argmax_tokens(solution_embedding)
        
        return RefinementResult(
            solution=final_tokens,
            reasoning_trace=reasoning_trace,
            final_confidence=confidence,
            num_steps=self.max_iterations,
            converged=False,
            total_time=time.time() - start_time
        )
    
    def generate(
        self,
        task_tokens: torch.Tensor,
        temperature: float = 1.0,
        return_trace: bool = False
    ) -> RefinementResult:
        """
        Generate solution with sampling
        
        Args:
            task_tokens: Input task specification [B, L]
            temperature: Sampling temperature
            return_trace: Whether to return reasoning trace
            
        Returns:
            RefinementResult with generated solution
        """
        # Run recursive refinement
        with torch.no_grad():
            result = self.forward(task_tokens, return_trace=return_trace)
        
        # Apply temperature sampling to final solution
        if temperature != 1.0:
            solution_embedding = self.output_embedding(result.solution)
            solution_logits = self.output_head(solution_embedding)
            sampled_tokens = self.output_head.generate_tokens(
                solution_embedding, temperature=temperature
            )
            result.solution = sampled_tokens
        
        return result