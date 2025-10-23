"""
Layer 2: Structural Rule Graph (K_R)

Implements explicit formal rules for domain-specific verification.
Enables zero-shot verification competence from initialization.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional, Tuple, Any
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass


@dataclass
class Rule:
    """A single formal rule in the knowledge graph"""
    id: str
    name: str
    description: str
    rule_type: str  # "syntax", "type", "constraint", "api"
    pattern: Optional[str] = None
    condition: Optional[str] = None
    violation_message: str = ""
    severity: str = "error"  # "error", "warning", "info"


@dataclass(frozen=True)
class Violation:
    """A rule violation detected during verification"""
    rule_id: str
    rule_name: str
    message: str
    severity: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class RuleVerifier(ABC):
    """Abstract base class for domain-specific rule verification"""
    
    @abstractmethod
    def verify(self, solution_tokens: List[int], task_tokens: List[int]) -> List[Violation]:
        """Verify solution against domain rules"""
        pass
    
    @abstractmethod
    def suggest_initial_solution(self, task_tokens: List[int]) -> Optional[torch.Tensor]:
        """Suggest initial solution based on rules"""
        pass
    
    @abstractmethod
    def embed_violations(self, violations: List[str]) -> Optional[torch.Tensor]:
        """Convert violations to embeddings for neural network"""
        pass


class StructuralRuleGraph(nn.Module):
    """
    Layer 2: Structural Rule Graph (K_R)
    
    Contains explicit formal rules for verification:
    - Grammar rules (G)
    - Type system rules (T) 
    - API specifications (A)
    
    Provides zero-shot verification without training.
    """
    
    def __init__(
        self,
        domain: str,
        rule_file: Optional[str] = None,
        embedding_dim: int = 512
    ):
        super().__init__()
        self.domain = domain
        self.embedding_dim = embedding_dim
        
        # Rule storage
        self.rules: Dict[str, Rule] = {}
        self.rule_categories: Dict[str, Set[str]] = {
            "syntax": set(),
            "type": set(), 
            "constraint": set(),
            "api": set()
        }
        
        # Verification components
        self.verifier: Optional[RuleVerifier] = None
        
        # Embeddings for rule violations
        self.violation_embedding = nn.Embedding(1000, embedding_dim)  # Up to 1000 violation types
        self.violation_to_id: Dict[str, int] = {}
        self.next_violation_id = 0
        
        # Load rules if file provided
        if rule_file:
            self.load_rules(rule_file)
    
    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge graph"""
        self.rules[rule.id] = rule
        self.rule_categories[rule.rule_type].add(rule.id)
    
    def add_rules(self, rules: List[Rule]):
        """Add multiple rules"""
        for rule in rules:
            self.add_rule(rule)
    
    def load_rules(self, rule_file: str):
        """Load rules from JSON file"""
        try:
            with open(rule_file, 'r') as f:
                rule_data = json.load(f)
            
            for rule_dict in rule_data.get('rules', []):
                rule = Rule(**rule_dict)
                self.add_rule(rule)
                
        except Exception as e:
            print(f"Warning: Could not load rules from {rule_file}: {e}")
    
    def save_rules(self, rule_file: str):
        """Save rules to JSON file"""
        rule_data = {
            'domain': self.domain,
            'rules': [
                {
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'rule_type': rule.rule_type,
                    'pattern': rule.pattern,
                    'condition': rule.condition,
                    'violation_message': rule.violation_message,
                    'severity': rule.severity
                }
                for rule in self.rules.values()
            ]
        }
        
        with open(rule_file, 'w') as f:
            json.dump(rule_data, f, indent=2)
    
    def set_verifier(self, verifier: RuleVerifier):
        """Set the domain-specific verifier"""
        self.verifier = verifier
    
    def verify(self, solution_tokens: List[int], task_tokens: List[int]) -> List[Violation]:
        """
        Verify solution against all rules
        
        Args:
            solution_tokens: Generated solution tokens
            task_tokens: Original task specification
            
        Returns:
            List of violations found
        """
        if self.verifier is None:
            return []
        
        return self.verifier.verify(solution_tokens, task_tokens)
    
    def suggest_initial_solution(self, task_tokens: List[int]) -> Optional[torch.Tensor]:
        """
        Suggest initial solution guided by rules
        
        Args:
            task_tokens: Task specification tokens
            
        Returns:
            Initial solution tokens if available
        """
        if self.verifier is None:
            return None
            
        return self.verifier.suggest_initial_solution(task_tokens)
    
    def embed_violations(self, violations: List[str]) -> Optional[torch.Tensor]:
        """
        Convert violations to neural network embeddings
        
        Args:
            violations: List of violation descriptions
            
        Returns:
            Violation embeddings [V, D] where V is number of violations
        """
        if not violations:
            return None
        
        violation_ids = []
        for violation in violations:
            if violation not in self.violation_to_id:
                if self.next_violation_id >= 1000:
                    # Reuse oldest violation ID
                    self.next_violation_id = 0
                self.violation_to_id[violation] = self.next_violation_id
                self.next_violation_id += 1
            
            violation_ids.append(self.violation_to_id[violation])
        
        violation_tensor = torch.tensor(violation_ids, dtype=torch.long)
        embeddings = self.violation_embedding(violation_tensor)
        
        return embeddings
    
    def get_rules_by_type(self, rule_type: str) -> List[Rule]:
        """Get all rules of a specific type"""
        rule_ids = self.rule_categories.get(rule_type, set())
        return [self.rules[rule_id] for rule_id in rule_ids]
    
    def get_rule_coverage(self) -> Dict[str, int]:
        """Get count of rules by category"""
        return {
            category: len(rule_ids) 
            for category, rule_ids in self.rule_categories.items()
        }
    
    def check_rule_completeness(self) -> Dict[str, bool]:
        """Check if essential rule categories are covered"""
        essential_categories = ["syntax", "type"]
        completeness = {}
        
        for category in essential_categories:
            completeness[category] = len(self.rule_categories[category]) > 0
        
        return completeness
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get statistics about violations encountered"""
        return {
            "total_violation_types": len(self.violation_to_id),
            "embedding_usage": self.next_violation_id / 1000.0,
            "most_common_violations": list(self.violation_to_id.keys())[:10]
        }
    
    def forward(self, violations: List[str]) -> torch.Tensor:
        """
        Forward pass for embedding violations
        
        Args:
            violations: List of violation descriptions
            
        Returns:
            Embedded violations [V, D]
        """
        return self.embed_violations(violations)


class MultiDomainRuleGraph(nn.Module):
    """
    Rule graph that supports multiple domains
    
    Allows switching between different domain rule sets
    (e.g., Python, SQL, Mathematics) within the same model.
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.domains: Dict[str, StructuralRuleGraph] = {}
        self.active_domain = None
    
    def add_domain(self, domain: str, rule_graph: StructuralRuleGraph):
        """Add a domain-specific rule graph"""
        self.domains[domain] = rule_graph
        
        # Register as submodule for proper parameter handling
        self.add_module(f"domain_{domain}", rule_graph)
    
    def set_active_domain(self, domain: str):
        """Set the currently active domain"""
        if domain not in self.domains:
            raise ValueError(f"Domain {domain} not found. Available: {list(self.domains.keys())}")
        self.active_domain = domain
    
    def get_active_graph(self) -> StructuralRuleGraph:
        """Get the currently active rule graph"""
        if self.active_domain is None:
            raise ValueError("No active domain set")
        return self.domains[self.active_domain]
    
    def verify(self, solution_tokens: List[int], task_tokens: List[int]) -> List[Violation]:
        """Verify using active domain rules"""
        return self.get_active_graph().verify(solution_tokens, task_tokens)
    
    def suggest_initial_solution(self, task_tokens: List[int]) -> Optional[torch.Tensor]:
        """Suggest initial solution using active domain"""
        return self.get_active_graph().suggest_initial_solution(task_tokens)
    
    def embed_violations(self, violations: List[str]) -> Optional[torch.Tensor]:
        """Embed violations using active domain"""
        return self.get_active_graph().embed_violations(violations)
    
    def get_domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all domains"""
        stats = {}
        for domain, rule_graph in self.domains.items():
            stats[domain] = {
                "rule_coverage": rule_graph.get_rule_coverage(),
                "completeness": rule_graph.check_rule_completeness(),
                "violation_stats": rule_graph.get_violation_stats()
            }
        return stats