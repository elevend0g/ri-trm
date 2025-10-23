"""
Python Domain Setup for RI-TRM

Sets up the complete Python domain with rules, facts, and initial path memory.
"""

import json
import os
from typing import Dict, List, Any, Optional

from ...knowledge.rule_graph import StructuralRuleGraph, Rule
from ...knowledge.fact_graph import FactualKnowledgeGraph, Entity, Fact
from ...knowledge.path_memory import PathMemoryGraph, ReasoningPath
from .rules import PythonRuleVerifier


class PythonDomainSetup:
    """
    Complete setup for Python domain in RI-TRM
    
    Initializes:
    - Layer 2 (K_R): Python syntax/type rules
    - Layer 1 (K_F): Python standard library facts  
    - Layer 3 (K_P): Initial debugging path memory
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        vocab_size: int = 32000,
        enable_type_checking: bool = True,
        strict_mode: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.enable_type_checking = enable_type_checking
        self.strict_mode = strict_mode
        
        # Knowledge components
        self.rule_graph: Optional[StructuralRuleGraph] = None
        self.fact_graph: Optional[FactualKnowledgeGraph] = None
        self.path_memory: Optional[PathMemoryGraph] = None
        self.rule_verifier: Optional[PythonRuleVerifier] = None
    
    def setup_complete_domain(self) -> Dict[str, Any]:
        """
        Set up complete Python domain with all knowledge layers
        
        Returns:
            Dictionary containing all initialized components
        """
        # Initialize components
        self.rule_graph = self._setup_rule_graph()
        self.fact_graph = self._setup_fact_graph()
        self.path_memory = self._setup_path_memory()
        self.rule_verifier = self._setup_rule_verifier()
        
        # Connect components
        self.rule_graph.set_verifier(self.rule_verifier)
        
        return {
            "rule_graph": self.rule_graph,
            "fact_graph": self.fact_graph,
            "path_memory": self.path_memory,
            "rule_verifier": self.rule_verifier,
            "domain": "python",
            "setup_complete": True
        }
    
    def _setup_rule_graph(self) -> StructuralRuleGraph:
        """Initialize Python structural rules (Layer 2)"""
        rule_graph = StructuralRuleGraph(
            domain="python",
            embedding_dim=self.embedding_dim
        )
        
        # Add comprehensive Python rules
        python_rules = self._get_python_rules()
        rule_graph.add_rules(python_rules)
        
        return rule_graph
    
    def _setup_fact_graph(self) -> FactualKnowledgeGraph:
        """Initialize Python factual knowledge (Layer 1)"""
        fact_graph = FactualKnowledgeGraph(
            domain="python",
            embedding_dim=self.embedding_dim
        )
        
        # Add Python standard library facts
        self._populate_python_facts(fact_graph)
        
        return fact_graph
    
    def _setup_path_memory(self) -> PathMemoryGraph:
        """Initialize path memory with Python debugging patterns (Layer 3)"""
        path_memory = PathMemoryGraph(
            embedding_dim=self.embedding_dim,
            max_paths=10000,  # Smaller for demo
            learning_rate=0.1,
            myelination_boost=1.1,
            decay_rate=0.95,
            myelination_threshold=5,  # Lower threshold for demo
            epsilon_init=0.3
        )
        
        # Add initial Python debugging paths
        self._populate_initial_paths(path_memory)
        
        return path_memory
    
    def _setup_rule_verifier(self) -> PythonRuleVerifier:
        """Initialize Python rule verifier"""
        return PythonRuleVerifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_type_checking=self.enable_type_checking,
            strict_mode=self.strict_mode
        )
    
    def _get_python_rules(self) -> List[Rule]:
        """Get comprehensive Python rule set"""
        return [
            # Syntax Rules
            Rule(
                id="py_syntax_valid",
                name="Valid Python Syntax",
                description="Code must be syntactically valid Python",
                rule_type="syntax",
                pattern="AST_PARSEABLE",
                violation_message="Syntax error: invalid Python syntax",
                severity="error"
            ),
            Rule(
                id="py_indentation_consistent",
                name="Consistent Indentation",
                description="Indentation must be consistent (4 spaces recommended)",
                rule_type="syntax",
                pattern="CONSISTENT_INDENT",
                violation_message="Inconsistent indentation",
                severity="error"
            ),
            Rule(
                id="py_colon_required",
                name="Colon Required",
                description="Control structures require colons",
                rule_type="syntax",
                pattern="COLON_AFTER_CONTROL",
                violation_message="Missing colon after if/for/while/def/class",
                severity="error"
            ),
            
            # Type System Rules
            Rule(
                id="py_function_return_type",
                name="Function Return Type",
                description="Functions returning values should have type hints",
                rule_type="type",
                pattern="RETURN_TYPE_HINT",
                violation_message="Function should have return type annotation",
                severity="warning"
            ),
            Rule(
                id="py_parameter_types",
                name="Parameter Type Hints",
                description="Function parameters should have type hints",
                rule_type="type",
                pattern="PARAM_TYPE_HINT",
                violation_message="Parameter should have type annotation",
                severity="info"
            ),
            Rule(
                id="py_variable_types",
                name="Variable Type Consistency",
                description="Variables should maintain consistent types",
                rule_type="type",
                pattern="TYPE_CONSISTENT",
                violation_message="Type inconsistency detected",
                severity="warning"
            ),
            
            # Constraint Rules
            Rule(
                id="py_import_valid",
                name="Valid Imports",
                description="All imports must reference existing modules",
                rule_type="constraint",
                pattern="VALID_IMPORT",
                violation_message="Import error: module not found",
                severity="error"
            ),
            Rule(
                id="py_variable_defined",
                name="Variable Definition",
                description="Variables must be defined before use",
                rule_type="constraint",
                pattern="VAR_DEFINED",
                violation_message="NameError: variable not defined",
                severity="error"
            ),
            Rule(
                id="py_function_exists",
                name="Function Existence",
                description="Called functions must be defined",
                rule_type="constraint",
                pattern="FUNC_DEFINED",
                violation_message="Function not defined",
                severity="error"
            ),
            
            # API Rules
            Rule(
                id="py_builtin_args",
                name="Built-in Function Arguments",
                description="Built-in functions must be called with correct arguments",
                rule_type="api",
                pattern="BUILTIN_ARGS_CORRECT",
                violation_message="Incorrect number of arguments",
                severity="error"
            ),
            Rule(
                id="py_method_exists",
                name="Method Existence",
                description="Called methods must exist on objects",
                rule_type="api",
                pattern="METHOD_EXISTS",
                violation_message="Method does not exist",
                severity="error"
            )
        ]
    
    def _populate_python_facts(self, fact_graph: FactualKnowledgeGraph):
        """Populate fact graph with Python standard library knowledge"""
        
        # Add standard library modules
        stdlib_modules = [
            ("os", "operating system interface"),
            ("sys", "system-specific parameters and functions"),
            ("json", "JSON encoder and decoder"),
            ("time", "time-related functions"),
            ("datetime", "date and time handling"),
            ("math", "mathematical functions"),
            ("random", "random number generation"),
            ("collections", "specialized container datatypes"),
            ("itertools", "functions creating iterators"),
            ("functools", "higher-order functions and operations on callable objects"),
            ("pathlib", "object-oriented filesystem paths"),
            ("re", "regular expression operations"),
            ("urllib", "URL handling modules"),
            ("http", "HTTP modules"),
            ("csv", "CSV file reading and writing"),
            ("sqlite3", "SQLite database interface")
        ]
        
        for module_name, description in stdlib_modules:
            # Add module entity
            entity = Entity(
                id=f"module_{module_name}",
                name=module_name,
                type="module",
                properties={"description": description, "stdlib": True}
            )
            fact_graph.add_entity(entity)
            
            # Add facts about module
            fact_graph.add_fact(Fact(
                subject=f"module_{module_name}",
                predicate="is_a",
                object="python_module",
                confidence=1.0
            ))
            
            fact_graph.add_fact(Fact(
                subject=f"module_{module_name}",
                predicate="part_of",
                object="python_stdlib",
                confidence=1.0
            ))
        
        # Add built-in functions
        builtins = [
            ("print", "output to console"),
            ("len", "returns length of object"),
            ("range", "generates sequence of numbers"),
            ("str", "converts to string"),
            ("int", "converts to integer"),
            ("float", "converts to float"),
            ("list", "creates list"),
            ("dict", "creates dictionary"),
            ("tuple", "creates tuple"),
            ("set", "creates set"),
            ("bool", "converts to boolean"),
            ("type", "returns object type"),
            ("isinstance", "checks if object is instance of class"),
            ("hasattr", "checks if object has attribute"),
            ("sum", "sums iterable"),
            ("max", "returns maximum"),
            ("min", "returns minimum"),
            ("abs", "returns absolute value"),
            ("round", "rounds number"),
            ("sorted", "returns sorted list"),
            ("enumerate", "returns enumerate object"),
            ("zip", "combines iterables"),
            ("map", "applies function to iterable"),
            ("filter", "filters iterable"),
            ("any", "returns True if any element is true"),
            ("all", "returns True if all elements are true")
        ]
        
        for func_name, description in builtins:
            entity = Entity(
                id=f"builtin_{func_name}",
                name=func_name,
                type="function",
                properties={"description": description, "builtin": True}
            )
            fact_graph.add_entity(entity)
            
            fact_graph.add_fact(Fact(
                subject=f"builtin_{func_name}",
                predicate="is_a",
                object="python_builtin",
                confidence=1.0
            ))
        
        # Add common third-party libraries
        third_party = [
            ("numpy", "numerical computing"),
            ("pandas", "data analysis and manipulation"),
            ("matplotlib", "plotting and visualization"),
            ("torch", "deep learning framework"),
            ("tensorflow", "machine learning platform"),
            ("requests", "HTTP library"),
            ("flask", "web framework"),
            ("django", "web framework"),
            ("sklearn", "machine learning library"),
            ("scipy", "scientific computing")
        ]
        
        for lib_name, description in third_party:
            entity = Entity(
                id=f"library_{lib_name}",
                name=lib_name,
                type="library",
                properties={"description": description, "third_party": True}
            )
            fact_graph.add_entity(entity)
            
            fact_graph.add_fact(Fact(
                subject=f"library_{lib_name}",
                predicate="is_a",
                object="python_library",
                confidence=0.9  # Slightly lower confidence for third-party
            ))
    
    def _populate_initial_paths(self, path_memory: PathMemoryGraph):
        """Populate path memory with common Python debugging patterns"""
        
        initial_paths = [
            # Syntax Error Fixes
            ReasoningPath(
                id="indent_fix_1",
                error_state="IndentationError at line 5",
                action="Add_4_spaces",
                result_state="Syntax_Valid",
                weight=0.95,
                usage_count=15,
                metadata={"pattern": "indentation", "fix_type": "add_spaces"}
            ),
            ReasoningPath(
                id="colon_fix_1",
                error_state="SyntaxError: missing colon",
                action="Add_colon_after_if",
                result_state="Syntax_Valid",
                weight=0.92,
                usage_count=12,
                metadata={"pattern": "missing_colon", "fix_type": "add_punctuation"}
            ),
            ReasoningPath(
                id="paren_fix_1",
                error_state="SyntaxError: unmatched parentheses",
                action="Add_closing_paren",
                result_state="Syntax_Valid",
                weight=0.88,
                usage_count=8,
                metadata={"pattern": "unmatched_paren", "fix_type": "balance_delimiters"}
            ),
            
            # Name/Import Error Fixes
            ReasoningPath(
                id="import_fix_1",
                error_state="NameError: 'np' not defined",
                action="Import_numpy_as_np",
                result_state="Code_Runs",
                weight=0.94,
                usage_count=20,
                metadata={"pattern": "missing_import", "fix_type": "add_import"}
            ),
            ReasoningPath(
                id="import_fix_2",
                error_state="NameError: 'pd' not defined",
                action="Import_pandas_as_pd",
                result_state="Code_Runs",
                weight=0.91,
                usage_count=16,
                metadata={"pattern": "missing_import", "fix_type": "add_import"}
            ),
            ReasoningPath(
                id="variable_fix_1",
                error_state="NameError: 'result' not defined",
                action="Initialize_variable_result",
                result_state="Code_Runs",
                weight=0.86,
                usage_count=10,
                metadata={"pattern": "undefined_variable", "fix_type": "initialize"}
            ),
            
            # Type Error Fixes
            ReasoningPath(
                id="type_fix_1",
                error_state="TypeError: cannot add string and integer",
                action="Convert_int_to_str",
                result_state="Code_Runs",
                weight=0.89,
                usage_count=14,
                metadata={"pattern": "type_mismatch", "fix_type": "type_conversion"}
            ),
            ReasoningPath(
                id="type_fix_2",
                error_state="TypeError: list indices must be integers",
                action="Convert_to_int",
                result_state="Code_Runs",
                weight=0.84,
                usage_count=7,
                metadata={"pattern": "index_type", "fix_type": "type_conversion"}
            ),
            
            # Attribute Error Fixes
            ReasoningPath(
                id="attr_fix_1",
                error_state="AttributeError: 'list' has no attribute 'append'",
                action="Fix_method_name_to_append",
                result_state="Code_Runs",
                weight=0.82,
                usage_count=6,
                metadata={"pattern": "wrong_method", "fix_type": "method_correction"}
            ),
            
            # Function Definition Fixes
            ReasoningPath(
                id="func_fix_1",
                error_state="Missing return statement in function",
                action="Add_return_statement",
                result_state="Tests_Pass",
                weight=0.90,
                usage_count=18,
                metadata={"pattern": "missing_return", "fix_type": "add_statement"}
            ),
            ReasoningPath(
                id="func_fix_2",
                error_state="Function missing required parameter",
                action="Add_parameter_to_function",
                result_state="Tests_Pass",
                weight=0.87,
                usage_count=11,
                metadata={"pattern": "missing_parameter", "fix_type": "add_parameter"}
            )
        ]
        
        # Add all initial paths
        for path in initial_paths:
            path_memory.add_path(path)
    
    def save_domain_setup(self, save_dir: str):
        """Save domain setup to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save rule graph
        if self.rule_graph:
            rule_file = os.path.join(save_dir, "python_rules.json")
            self.rule_graph.save_rules(rule_file)
        
        # Save path memory
        if self.path_memory:
            path_file = os.path.join(save_dir, "python_paths.json")
            self.path_memory.save_memory(path_file)
        
        # Save setup metadata
        metadata = {
            "domain": "python",
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "enable_type_checking": self.enable_type_checking,
            "strict_mode": self.strict_mode,
            "setup_complete": True
        }
        
        metadata_file = os.path.join(save_dir, "domain_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_domain_setup(self, save_dir: str) -> Dict[str, Any]:
        """Load domain setup from files"""
        # Load metadata
        metadata_file = os.path.join(save_dir, "domain_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.embedding_dim = metadata.get("embedding_dim", 512)
            self.vocab_size = metadata.get("vocab_size", 32000)
            self.enable_type_checking = metadata.get("enable_type_checking", True)
            self.strict_mode = metadata.get("strict_mode", False)
        
        # Setup components
        domain_components = self.setup_complete_domain()
        
        # Load saved data
        rule_file = os.path.join(save_dir, "python_rules.json")
        if os.path.exists(rule_file) and self.rule_graph:
            self.rule_graph.load_rules(rule_file)
        
        path_file = os.path.join(save_dir, "python_paths.json")
        if os.path.exists(path_file) and self.path_memory:
            self.path_memory.load_memory(path_file)
        
        return domain_components
    
    def get_setup_statistics(self) -> Dict[str, Any]:
        """Get statistics about the domain setup"""
        stats = {
            "domain": "python",
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size
        }
        
        if self.rule_graph:
            stats["rule_coverage"] = self.rule_graph.get_rule_coverage()
            stats["rule_completeness"] = self.rule_graph.check_rule_completeness()
        
        if self.fact_graph:
            stats["knowledge_stats"] = self.fact_graph.get_knowledge_stats()
        
        if self.path_memory:
            stats["path_stats"] = self.path_memory.get_path_statistics()
        
        return stats