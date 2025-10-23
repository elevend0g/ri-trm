"""
Evaluation Metrics for RI-TRM

Comprehensive metrics for measuring:
- Performance (accuracy, pass rates)
- Efficiency (training time, parameters, inference speed)
- Interpretability (reasoning trace quality)
"""

import torch
import time
import psutil
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    """Core performance evaluation metrics"""
    # Task success metrics
    task_success_rate: float
    test_pass_rate: float
    syntax_correctness: float
    semantic_correctness: float
    
    # Confidence calibration
    confidence_accuracy: float
    confidence_correlation: float
    
    # Reasoning quality
    avg_reasoning_steps: float
    reasoning_stability: float
    early_stopping_rate: float
    
    # Error analysis
    syntax_error_rate: float
    runtime_error_rate: float
    logic_error_rate: float
    
    # Overall score
    composite_score: float


@dataclass
class EfficiencyMetrics:
    """Efficiency and resource usage metrics"""
    # Model size
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    
    # Training efficiency
    training_time_hours: float
    training_samples: int
    samples_per_second: float
    convergence_epoch: int
    
    # Inference efficiency
    avg_inference_time_ms: float
    avg_reasoning_steps: float
    memory_usage_mb: float
    
    # Comparison metrics
    efficiency_vs_baseline: float
    parameter_reduction: float
    speed_improvement: float


@dataclass
class InterpretabilityMetrics:
    """Interpretability and explainability metrics"""
    # Path trace quality
    trace_completeness: float
    trace_coherence: float
    decision_clarity: float
    
    # Path memory insights
    path_diversity: float
    path_confidence_correlation: float
    successful_path_reuse_rate: float
    
    # Human understandability
    explanation_clarity_score: float
    debugging_utility_score: float
    
    # Rule utilization
    rule_coverage: float
    rule_effectiveness: float


class MetricsCalculator:
    """Calculator for RI-TRM evaluation metrics"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset metrics collection"""
        self.task_results = []
        self.timing_results = []
        self.reasoning_traces = []
        self.confidence_scores = []
        self.ground_truth_success = []
        self.memory_usage = []
    
    def add_task_result(
        self,
        task_id: str,
        predicted_solution: torch.Tensor,
        ground_truth: Optional[torch.Tensor],
        test_results: Dict[str, Any],
        reasoning_trace: List[Any],
        confidence_score: float,
        inference_time: float,
        memory_usage: float
    ):
        """Add results from a single task evaluation"""
        
        result = {
            "task_id": task_id,
            "predicted_solution": predicted_solution,
            "ground_truth": ground_truth,
            "test_results": test_results,
            "reasoning_trace": reasoning_trace,
            "confidence_score": confidence_score,
            "inference_time": inference_time,
            "memory_usage": memory_usage,
            "success": test_results.get("passed_tests", 0) == test_results.get("total_tests", 1)
        }
        
        self.task_results.append(result)
        self.reasoning_traces.append(reasoning_trace)
        self.confidence_scores.append(confidence_score)
        self.ground_truth_success.append(result["success"])
        self.timing_results.append(inference_time)
        self.memory_usage.append(memory_usage)
    
    def calculate_performance_metrics(self) -> EvaluationMetrics:
        """Calculate core performance metrics"""
        if not self.task_results:
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Task success metrics
        task_success_rate = sum(r["success"] for r in self.task_results) / len(self.task_results)
        
        test_pass_rates = []
        syntax_correct = 0
        semantic_correct = 0
        
        for result in self.task_results:
            test_results = result["test_results"]
            if test_results.get("total_tests", 0) > 0:
                pass_rate = test_results.get("passed_tests", 0) / test_results["total_tests"]
                test_pass_rates.append(pass_rate)
            
            # Syntax correctness (simplified check)
            if "syntax_error" not in test_results.get("error", ""):
                syntax_correct += 1
            
            # Semantic correctness (if tests pass)
            if result["success"]:
                semantic_correct += 1
        
        test_pass_rate = np.mean(test_pass_rates) if test_pass_rates else 0.0
        syntax_correctness = syntax_correct / len(self.task_results)
        semantic_correctness = semantic_correct / len(self.task_results)
        
        # Confidence calibration
        confidence_accuracy = self._calculate_confidence_accuracy()
        confidence_correlation = self._calculate_confidence_correlation()
        
        # Reasoning quality
        avg_reasoning_steps = np.mean([len(trace) for trace in self.reasoning_traces])
        reasoning_stability = self._calculate_reasoning_stability()
        early_stopping_rate = self._calculate_early_stopping_rate()
        
        # Error analysis
        error_rates = self._analyze_error_types()
        
        # Composite score (weighted combination)
        composite_score = (
            0.4 * task_success_rate +
            0.3 * test_pass_rate +
            0.2 * syntax_correctness +
            0.1 * confidence_accuracy
        )
        
        return EvaluationMetrics(
            task_success_rate=task_success_rate,
            test_pass_rate=test_pass_rate,
            syntax_correctness=syntax_correctness,
            semantic_correctness=semantic_correctness,
            confidence_accuracy=confidence_accuracy,
            confidence_correlation=confidence_correlation,
            avg_reasoning_steps=avg_reasoning_steps,
            reasoning_stability=reasoning_stability,
            early_stopping_rate=early_stopping_rate,
            syntax_error_rate=error_rates["syntax"],
            runtime_error_rate=error_rates["runtime"],
            logic_error_rate=error_rates["logic"],
            composite_score=composite_score
        )
    
    def calculate_efficiency_metrics(
        self,
        model,
        training_time_hours: float,
        training_samples: int,
        convergence_epoch: int
    ) -> EfficiencyMetrics:
        """Calculate efficiency metrics"""
        
        # Model size metrics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        # Training efficiency
        samples_per_second = training_samples / (training_time_hours * 3600) if training_time_hours > 0 else 0
        
        # Inference efficiency
        avg_inference_time = np.mean(self.timing_results) if self.timing_results else 0
        avg_memory_usage = np.mean(self.memory_usage) if self.memory_usage else 0
        avg_reasoning_steps = np.mean([len(trace) for trace in self.reasoning_traces])
        
        return EfficiencyMetrics(
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            training_time_hours=training_time_hours,
            training_samples=training_samples,
            samples_per_second=samples_per_second,
            convergence_epoch=convergence_epoch,
            avg_inference_time_ms=avg_inference_time * 1000,
            avg_reasoning_steps=avg_reasoning_steps,
            memory_usage_mb=avg_memory_usage,
            efficiency_vs_baseline=1.0,  # Placeholder
            parameter_reduction=1.0,     # Placeholder  
            speed_improvement=1.0        # Placeholder
        )
    
    def calculate_interpretability_metrics(self, path_memory=None) -> InterpretabilityMetrics:
        """Calculate interpretability metrics"""
        
        # Path trace quality
        trace_completeness = self._calculate_trace_completeness()
        trace_coherence = self._calculate_trace_coherence()
        decision_clarity = self._calculate_decision_clarity()
        
        # Path memory insights
        path_metrics = self._analyze_path_memory(path_memory) if path_memory else {}
        
        # Human understandability (simplified scoring)
        explanation_clarity = self._score_explanation_clarity()
        debugging_utility = self._score_debugging_utility()
        
        # Rule utilization
        rule_coverage, rule_effectiveness = self._analyze_rule_usage()
        
        return InterpretabilityMetrics(
            trace_completeness=trace_completeness,
            trace_coherence=trace_coherence,
            decision_clarity=decision_clarity,
            path_diversity=path_metrics.get("diversity", 0.0),
            path_confidence_correlation=path_metrics.get("confidence_correlation", 0.0),
            successful_path_reuse_rate=path_metrics.get("reuse_rate", 0.0),
            explanation_clarity_score=explanation_clarity,
            debugging_utility_score=debugging_utility,
            rule_coverage=rule_coverage,
            rule_effectiveness=rule_effectiveness
        )
    
    def _calculate_confidence_accuracy(self) -> float:
        """Calculate how well confidence scores match actual success"""
        if not self.confidence_scores or not self.ground_truth_success:
            return 0.0
        
        # Binary classification accuracy
        predicted_success = [c > 0.5 for c in self.confidence_scores]
        correct_predictions = sum(
            p == g for p, g in zip(predicted_success, self.ground_truth_success)
        )
        
        return correct_predictions / len(self.confidence_scores)
    
    def _calculate_confidence_correlation(self) -> float:
        """Calculate correlation between confidence and success rate"""
        if len(self.confidence_scores) < 2:
            return 0.0
        
        return np.corrcoef(self.confidence_scores, self.ground_truth_success)[0, 1]
    
    def _calculate_reasoning_stability(self) -> float:
        """Measure how stable reasoning traces are across similar problems"""
        if len(self.reasoning_traces) < 2:
            return 1.0
        
        # Simplified: measure variance in trace lengths
        trace_lengths = [len(trace) for trace in self.reasoning_traces]
        stability = 1.0 - (np.std(trace_lengths) / np.mean(trace_lengths))
        
        return max(0.0, stability)
    
    def _calculate_early_stopping_rate(self) -> float:
        """Calculate rate of early stopping vs. max iterations"""
        if not self.task_results:
            return 0.0
        
        early_stops = 0
        for result in self.task_results:
            trace = result["reasoning_trace"]
            if trace and hasattr(trace[-1], "step") and trace[-1].step < 15:  # Assuming max 16
                early_stops += 1
        
        return early_stops / len(self.task_results)
    
    def _analyze_error_types(self) -> Dict[str, float]:
        """Analyze distribution of error types"""
        error_counts = {"syntax": 0, "runtime": 0, "logic": 0}
        
        for result in self.task_results:
            test_results = result["test_results"]
            error_msg = test_results.get("error", "").lower()
            
            if "syntax" in error_msg:
                error_counts["syntax"] += 1
            elif "runtime" in error_msg or "exception" in error_msg:
                error_counts["runtime"] += 1
            elif not result["success"]:
                error_counts["logic"] += 1
        
        total = len(self.task_results)
        return {k: v / total for k, v in error_counts.items()}
    
    def _calculate_trace_completeness(self) -> float:
        """Measure how complete reasoning traces are"""
        if not self.reasoning_traces:
            return 0.0
        
        complete_traces = 0
        for trace in self.reasoning_traces:
            # Check if trace has key components
            has_reasoning = len(trace) > 0
            has_decisions = any(hasattr(step, "selected_path") for step in trace)
            
            if has_reasoning and has_decisions:
                complete_traces += 1
        
        return complete_traces / len(self.reasoning_traces)
    
    def _calculate_trace_coherence(self) -> float:
        """Measure logical coherence of reasoning traces"""
        if not self.reasoning_traces:
            return 0.0
        
        coherent_traces = 0
        for trace in self.reasoning_traces:
            if len(trace) < 2:
                coherent_traces += 1  # Trivially coherent
                continue
            
            # Check if consecutive steps make logical sense
            coherent = True
            for i in range(1, len(trace)):
                # Simplified: check if violations decrease
                if (hasattr(trace[i-1], "violations") and hasattr(trace[i], "violations")):
                    if len(trace[i].violations) > len(trace[i-1].violations):
                        coherent = False
                        break
            
            if coherent:
                coherent_traces += 1
        
        return coherent_traces / len(self.reasoning_traces)
    
    def _calculate_decision_clarity(self) -> float:
        """Measure how clear path selection decisions are"""
        if not self.reasoning_traces:
            return 0.0
        
        clear_decisions = 0
        total_decisions = 0
        
        for trace in self.reasoning_traces:
            for step in trace:
                if hasattr(step, "path_weight"):
                    total_decisions += 1
                    # High weight indicates clear decision
                    if step.path_weight > 0.7:
                        clear_decisions += 1
        
        return clear_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _analyze_path_memory(self, path_memory) -> Dict[str, float]:
        """Analyze path memory utilization and effectiveness"""
        if not path_memory:
            return {}
        
        stats = path_memory.get_path_statistics()
        
        # Path diversity (number of unique error types)
        diversity = min(1.0, stats.get("unique_error_states", 0) / 50.0)
        
        # Confidence correlation (how well path weights predict success)
        # Simplified placeholder
        confidence_correlation = 0.7
        
        # Reuse rate (how often paths are reused)
        total_usage = sum(path.usage_count for path in path_memory.paths.values())
        unique_paths = len(path_memory.paths)
        reuse_rate = (total_usage - unique_paths) / max(total_usage, 1)
        
        return {
            "diversity": diversity,
            "confidence_correlation": confidence_correlation,
            "reuse_rate": reuse_rate
        }
    
    def _score_explanation_clarity(self) -> float:
        """Score how clear the explanations are (simplified)"""
        if not self.reasoning_traces:
            return 0.0
        
        # Simplified: based on trace completeness and decision clarity
        completeness = self._calculate_trace_completeness()
        clarity = self._calculate_decision_clarity()
        
        return (completeness + clarity) / 2
    
    def _score_debugging_utility(self) -> float:
        """Score how useful traces are for debugging (simplified)"""
        if not self.reasoning_traces:
            return 0.0
        
        # Simplified: based on error localization and suggestions
        useful_traces = 0
        for trace in self.reasoning_traces:
            has_error_info = any(hasattr(step, "violations") and step.violations 
                               for step in trace)
            has_suggestions = any(hasattr(step, "selected_path") and step.selected_path
                                for step in trace)
            
            if has_error_info and has_suggestions:
                useful_traces += 1
        
        return useful_traces / len(self.reasoning_traces)
    
    def _analyze_rule_usage(self) -> Tuple[float, float]:
        """Analyze how well rules are utilized"""
        # Simplified placeholder
        # In practice, would analyze which rules are triggered and how effective they are
        
        rule_coverage = 0.8  # 80% of rules used
        rule_effectiveness = 0.7  # 70% of rule applications successful
        
        return rule_coverage, rule_effectiveness
    
    def generate_report(
        self,
        performance: EvaluationMetrics,
        efficiency: EfficiencyMetrics,
        interpretability: InterpretabilityMetrics
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        return {
            "summary": {
                "overall_score": performance.composite_score,
                "task_success_rate": performance.task_success_rate,
                "parameter_count": efficiency.total_parameters,
                "training_efficiency": efficiency.samples_per_second,
                "interpretability_score": (
                    interpretability.trace_completeness + 
                    interpretability.explanation_clarity_score
                ) / 2
            },
            "performance": {
                "task_success_rate": performance.task_success_rate,
                "test_pass_rate": performance.test_pass_rate,
                "syntax_correctness": performance.syntax_correctness,
                "semantic_correctness": performance.semantic_correctness,
                "confidence_accuracy": performance.confidence_accuracy,
                "avg_reasoning_steps": performance.avg_reasoning_steps
            },
            "efficiency": {
                "model_size_mb": efficiency.model_size_mb,
                "total_parameters": efficiency.total_parameters,
                "training_time_hours": efficiency.training_time_hours,
                "avg_inference_time_ms": efficiency.avg_inference_time_ms,
                "memory_usage_mb": efficiency.memory_usage_mb
            },
            "interpretability": {
                "trace_completeness": interpretability.trace_completeness,
                "trace_coherence": interpretability.trace_coherence,
                "decision_clarity": interpretability.decision_clarity,
                "explanation_clarity_score": interpretability.explanation_clarity_score,
                "debugging_utility_score": interpretability.debugging_utility_score
            },
            "detailed_results": [
                {
                    "task_id": result["task_id"],
                    "success": result["success"],
                    "confidence": result["confidence_score"],
                    "inference_time": result["inference_time"],
                    "reasoning_steps": len(result["reasoning_trace"])
                }
                for result in self.task_results
            ]
        }