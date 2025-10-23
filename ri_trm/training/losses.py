"""
Loss Functions for RI-TRM Training

Implements specialized losses for task-based training:
- Task completion loss
- Path consistency loss  
- Test-based loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class LossComponents:
    """Container for different loss components"""
    task_loss: torch.Tensor
    test_loss: torch.Tensor
    path_loss: torch.Tensor
    confidence_loss: torch.Tensor
    total_loss: torch.Tensor


class TaskLoss(nn.Module):
    """
    Task completion loss for RI-TRM
    
    Measures how well the model generates correct solutions
    compared to ground truth, when available.
    """
    
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int = 0,  # Padding token
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        # Standard cross-entropy loss with label smoothing
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        predicted_logits: torch.Tensor,  # [B, L, V]
        target_tokens: torch.Tensor,     # [B, L]
        mask: Optional[torch.Tensor] = None  # [B, L]
    ) -> torch.Tensor:
        """
        Compute task completion loss
        
        Args:
            predicted_logits: Model output logits
            target_tokens: Ground truth token sequence
            mask: Optional mask for valid positions
            
        Returns:
            Task loss value
        """
        batch_size, seq_len, vocab_size = predicted_logits.shape
        
        # Reshape for cross-entropy
        logits_flat = predicted_logits.view(-1, vocab_size)
        targets_flat = target_tokens.view(-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        # Compute cross-entropy loss
        loss = self.cross_entropy(logits_flat, targets_flat)
        
        return loss


class TestLoss(nn.Module):
    """
    Test-based loss for RI-TRM
    
    Measures how well generated solutions pass their test cases.
    This is a key innovation of RI-TRM - training on task success
    rather than just token prediction.
    """
    
    def __init__(self, test_weight: float = 1.0):
        super().__init__()
        self.test_weight = test_weight
    
    def forward(
        self,
        test_results: List[Dict[str, Any]],
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute test-based loss
        
        Args:
            test_results: List of test results for each item in batch
            batch_size: Size of the batch
            
        Returns:
            Test loss (0.0 if all tests pass, 1.0 if all fail)
        """
        if not test_results:
            return torch.tensor(0.0, requires_grad=True)
        
        total_pass_rate = 0.0
        
        for result in test_results:
            if result and "passed_tests" in result and "total_tests" in result:
                if result["total_tests"] > 0:
                    pass_rate = result["passed_tests"] / result["total_tests"]
                    total_pass_rate += pass_rate
                else:
                    total_pass_rate += 1.0  # No tests means no failure
            else:
                # If test execution failed, penalize heavily
                total_pass_rate += 0.0
        
        # Average pass rate across batch
        avg_pass_rate = total_pass_rate / batch_size
        
        # Loss is 1 - pass_rate (lower is better)
        test_loss = (1.0 - avg_pass_rate) * self.test_weight
        
        return torch.tensor(test_loss, requires_grad=True)


class PathConsistencyLoss(nn.Module):
    """
    Path consistency loss for RI-TRM
    
    Encourages the model to follow stable, consistent reasoning paths.
    Prevents the reasoning from being too chaotic or unstable.
    """
    
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        reasoning_trace: List[Any],
        target_consistency: float = 0.8
    ) -> torch.Tensor:
        """
        Compute path consistency loss
        
        Args:
            reasoning_trace: List of reasoning steps
            target_consistency: Target consistency score
            
        Returns:
            Consistency loss
        """
        if not reasoning_trace or len(reasoning_trace) < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # Measure consistency as similarity between consecutive reasoning steps
        consistencies = []
        
        for i in range(1, len(reasoning_trace)):
            step_1 = reasoning_trace[i-1]
            step_2 = reasoning_trace[i]
            
            # Extract latent states if available
            if hasattr(step_1, 'latent_state') and hasattr(step_2, 'latent_state'):
                state_1 = step_1.latent_state
                state_2 = step_2.latent_state
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    state_1.view(-1), 
                    state_2.view(-1), 
                    dim=0
                )
                consistencies.append(similarity)
        
        if not consistencies:
            return torch.tensor(0.0, requires_grad=True)
        
        # Average consistency
        avg_consistency = torch.stack(consistencies).mean()
        
        # Loss penalizes deviation from target consistency
        consistency_loss = F.mse_loss(avg_consistency, torch.tensor(target_consistency))
        
        return consistency_loss * self.consistency_weight


class ConfidenceLoss(nn.Module):
    """
    Confidence calibration loss
    
    Ensures the model's confidence scores are well-calibrated
    with actual performance.
    """
    
    def __init__(self, calibration_weight: float = 0.1):
        super().__init__()
        self.calibration_weight = calibration_weight
    
    def forward(
        self,
        confidence_scores: torch.Tensor,  # [B, L, 1] or [B, 1]
        success_indicators: torch.Tensor  # [B] binary success
    ) -> torch.Tensor:
        """
        Compute confidence calibration loss
        
        Args:
            confidence_scores: Model confidence predictions
            success_indicators: Binary indicators of task success
            
        Returns:
            Calibration loss
        """
        # Get sequence-level confidence (average if per-token)
        if confidence_scores.dim() == 3:
            seq_confidence = confidence_scores.mean(dim=1).squeeze(-1)  # [B]
        else:
            seq_confidence = confidence_scores.squeeze(-1)  # [B]
        
        # Binary cross-entropy between confidence and success
        success_float = success_indicators.float()
        calibration_loss = F.binary_cross_entropy(seq_confidence, success_float)
        
        return calibration_loss * self.calibration_weight


class RITRMLoss(nn.Module):
    """
    Combined loss function for RI-TRM training
    
    Integrates all loss components:
    - Task completion (when ground truth available)
    - Test success (key innovation)
    - Path consistency
    - Confidence calibration
    """
    
    def __init__(
        self,
        vocab_size: int,
        task_weight: float = 1.0,
        test_weight: float = 1.0,
        path_weight: float = 0.1,
        confidence_weight: float = 0.1,
        ignore_index: int = 0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.task_weight = task_weight
        self.test_weight = test_weight
        self.path_weight = path_weight
        self.confidence_weight = confidence_weight
        
        # Individual loss components
        self.task_loss = TaskLoss(vocab_size, ignore_index, label_smoothing)
        self.test_loss = TestLoss(test_weight)
        self.path_loss = PathConsistencyLoss(path_weight)
        self.confidence_loss = ConfidenceLoss(confidence_weight)
    
    def forward(
        self,
        predicted_logits: Optional[torch.Tensor] = None,  # [B, L, V]
        target_tokens: Optional[torch.Tensor] = None,     # [B, L]
        confidence_scores: Optional[torch.Tensor] = None, # [B, L, 1]
        test_results: Optional[List[Dict[str, Any]]] = None,
        reasoning_traces: Optional[List[Any]] = None,
        batch_size: int = 1,
        mask: Optional[torch.Tensor] = None
    ) -> LossComponents:
        """
        Compute combined RI-TRM loss
        
        Args:
            predicted_logits: Model output logits
            target_tokens: Ground truth tokens (if available)
            confidence_scores: Model confidence predictions
            test_results: Test execution results
            reasoning_traces: Reasoning step traces
            batch_size: Batch size
            mask: Token mask
            
        Returns:
            LossComponents with individual and total losses
        """
        losses = {}
        
        # 1. Task completion loss (if ground truth available)
        if predicted_logits is not None and target_tokens is not None:
            task_loss_val = self.task_loss(predicted_logits, target_tokens, mask)
            losses["task"] = task_loss_val * self.task_weight
        else:
            losses["task"] = torch.tensor(0.0, requires_grad=True)
        
        # 2. Test-based loss (key for RI-TRM)
        test_loss_val = self.test_loss(test_results or [], batch_size)
        losses["test"] = test_loss_val * self.test_weight
        
        # 3. Path consistency loss
        if reasoning_traces:
            path_loss_val = torch.stack([
                self.path_loss(trace) for trace in reasoning_traces
            ]).mean()
        else:
            path_loss_val = torch.tensor(0.0, requires_grad=True)
        losses["path"] = path_loss_val * self.path_weight
        
        # 4. Confidence calibration loss
        if confidence_scores is not None and test_results:
            # Convert test results to success indicators
            success_indicators = []
            for result in test_results:
                if result and "passed_tests" in result and "total_tests" in result:
                    success = 1.0 if result["passed_tests"] == result["total_tests"] else 0.0
                else:
                    success = 0.0
                success_indicators.append(success)
            
            if success_indicators:
                success_tensor = torch.tensor(success_indicators, device=confidence_scores.device)
                conf_loss_val = self.confidence_loss(confidence_scores, success_tensor)
                losses["confidence"] = conf_loss_val * self.confidence_weight
            else:
                losses["confidence"] = torch.tensor(0.0, requires_grad=True)
        else:
            losses["confidence"] = torch.tensor(0.0, requires_grad=True)
        
        # 5. Total loss
        total_loss = sum(losses.values())
        
        return LossComponents(
            task_loss=losses["task"],
            test_loss=losses["test"],
            path_loss=losses["path"],
            confidence_loss=losses["confidence"],
            total_loss=total_loss
        )
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {
            "task_weight": self.task_weight,
            "test_weight": self.test_weight,
            "path_weight": self.path_weight,
            "confidence_weight": self.confidence_weight
        }
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights during training"""
        if "task_weight" in new_weights:
            self.task_weight = new_weights["task_weight"]
        if "test_weight" in new_weights:
            self.test_weight = new_weights["test_weight"]
            self.test_loss.test_weight = new_weights["test_weight"]
        if "path_weight" in new_weights:
            self.path_weight = new_weights["path_weight"]
            self.path_loss.consistency_weight = new_weights["path_weight"]
        if "confidence_weight" in new_weights:
            self.confidence_weight = new_weights["confidence_weight"]
            self.confidence_loss.calibration_weight = new_weights["confidence_weight"]


class AdaptiveLossScheduler:
    """
    Adaptive loss weight scheduler for RI-TRM
    
    Adjusts loss weights during training based on performance.
    For example, increases test_weight when model is overfitting
    to token prediction but failing tests.
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.1,
        patience: int = 5
    ):
        self.current_weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.patience = patience
        
        # Tracking for adaptation
        self.performance_history = []
        self.steps_without_improvement = 0
    
    def step(
        self,
        loss_components: LossComponents,
        test_pass_rate: float,
        epoch: int
    ) -> Dict[str, float]:
        """
        Update loss weights based on current performance
        
        Args:
            loss_components: Current loss values
            test_pass_rate: Current test pass rate
            epoch: Current epoch number
            
        Returns:
            Updated loss weights
        """
        # Record performance
        performance = {
            "epoch": epoch,
            "test_pass_rate": test_pass_rate,
            "task_loss": loss_components.task_loss.item(),
            "test_loss": loss_components.test_loss.item()
        }
        self.performance_history.append(performance)
        
        # Check if we need to adapt weights
        if len(self.performance_history) >= 2:
            current_perf = self.performance_history[-1]
            previous_perf = self.performance_history[-2]
            
            # If test pass rate is declining while task loss improves
            # (indicates overfitting to tokens), increase test weight
            if (current_perf["test_pass_rate"] < previous_perf["test_pass_rate"] and
                current_perf["task_loss"] < previous_perf["task_loss"]):
                
                self.current_weights["test_weight"] *= (1 + self.adaptation_rate)
                self.current_weights["task_weight"] *= (1 - self.adaptation_rate * 0.5)
                
            # If both test and task performance are improving, no change needed
            elif (current_perf["test_pass_rate"] >= previous_perf["test_pass_rate"] and
                  current_perf["task_loss"] <= previous_perf["task_loss"]):
                
                # Gradually reduce adaptation (stable performance)
                pass
            
            # If performance is stagnating, increase path consistency weight
            # to encourage more stable reasoning
            else:
                self.steps_without_improvement += 1
                if self.steps_without_improvement >= self.patience:
                    self.current_weights["path_weight"] *= (1 + self.adaptation_rate)
                    self.steps_without_improvement = 0
        
        # Ensure weights stay in reasonable bounds
        self.current_weights = {
            k: max(0.01, min(10.0, v)) for k, v in self.current_weights.items()
        }
        
        return self.current_weights.copy()