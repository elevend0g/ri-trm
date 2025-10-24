"""
Python Domain Rule Verifier

Implements structural rules (K_R) for Python code generation:
- Python AST grammar rules
- Type system constraints
- Standard library API signatures
- Import validation
"""

import ast
import sys
import tokenize
from io import StringIO
from typing import List, Dict, Set, Optional, Tuple, Any
import re
import torch
import torch.nn as nn

from ...knowledge.rule_graph import RuleVerifier, Rule, Violation
from .verifier import PythonSyntaxVerifier, PythonTypeVerifier


class PythonRuleVerifier(RuleVerifier):
    """
    Python-specific rule verification for RI-TRM
    
    Implements Layer 2 (K_R) verification for Python code:
    1. Syntax validation using AST
    2. Basic type checking
    3. Import validation
    4. Standard library API checks
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 512,
        enable_type_checking: bool = True,
        enable_import_checking: bool = True,
        strict_mode: bool = False,
        tokenizer_name: str = "gpt2"
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enable_type_checking = enable_type_checking
        self.enable_import_checking = enable_import_checking
        self.strict_mode = strict_mode

        # Initialize specialized verifiers
        self.syntax_verifier = PythonSyntaxVerifier()

        if enable_type_checking:
            self.type_verifier = PythonTypeVerifier()
        else:
            self.type_verifier = None

        # Common Python patterns and rules
        self._init_python_rules()

        # Real tokenizer integration
        from ...tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer(model_name=tokenizer_name, max_length=512)
        
        # Standard library and common imports
        self.stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator',
            'pathlib', 're', 'urllib', 'http', 'csv', 'sqlite3'
        }
        
        # Common third-party libraries
        self.common_libraries = {
            'numpy', 'pandas', 'matplotlib', 'torch', 'tensorflow',
            'requests', 'flask', 'django', 'sklearn', 'scipy'
        }
    
    def _init_python_rules(self):
        """Initialize Python-specific rules"""
        self.python_rules = [
            Rule(
                id="py_syntax_valid",
                name="Valid Python Syntax",
                description="Code must be syntactically valid Python",
                rule_type="syntax",
                pattern="AST_PARSEABLE",
                violation_message="Syntax error: invalid Python syntax"
            ),
            Rule(
                id="py_indentation_consistent",
                name="Consistent Indentation",
                description="Indentation must be consistent (4 spaces recommended)",
                rule_type="syntax",
                pattern="CONSISTENT_INDENT",
                violation_message="Inconsistent indentation"
            ),
            Rule(
                id="py_function_return_type",
                name="Function Return Type",
                description="Functions that return values should have return type hints",
                rule_type="type",
                pattern="RETURN_TYPE_HINT",
                violation_message="Function should have return type annotation"
            ),
            Rule(
                id="py_import_valid",
                name="Valid Imports",
                description="All imports must reference existing modules",
                rule_type="constraint",
                pattern="VALID_IMPORT",
                violation_message="Import error: module not found"
            ),
            Rule(
                id="py_variable_defined",
                name="Variable Definition",
                description="Variables must be defined before use",
                rule_type="constraint",
                pattern="VAR_DEFINED",
                violation_message="NameError: variable not defined"
            )
        ]
    
    def _tokens_to_code(self, tokens: List[int]) -> str:
        """Convert token IDs to Python code string using real tokenizer"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _code_to_tokens(self, code: str) -> List[int]:
        """Convert Python code string to token IDs using real tokenizer"""
        return self.tokenizer.encode(code, padding=True, return_tensors=False)

    def _extract_task_requirements(self, task_tokens: List[int]) -> Dict[str, Any]:
        """Extract requirements from task specification"""
        # Placeholder implementation
        # In a real system, this would parse the task description
        return {
            "function_name": "solve_task",
            "parameters": [],
            "return_type": "Any",
            "imports_needed": [],
            "constraints": []
        }
    
    def verify(self, solution_tokens: List[int], task_tokens: List[int]) -> List[Violation]:
        """
        Verify Python code solution against structural rules
        
        Args:
            solution_tokens: Generated Python code as token IDs
            task_tokens: Original task specification as token IDs
            
        Returns:
            List of rule violations found
        """
        violations = []
        
        # Convert tokens to code
        code = self._tokens_to_code(solution_tokens)
        task_requirements = self._extract_task_requirements(task_tokens)
        
        # 1. Syntax verification
        syntax_violations = self.syntax_verifier.verify_syntax(code)
        violations.extend(syntax_violations)
        
        # If syntax is invalid, can't proceed with other checks
        if syntax_violations:
            return violations
        
        # 2. Parse AST for further analysis
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(Violation(
                rule_id="py_syntax_valid",
                rule_name="Valid Python Syntax",
                message=f"Syntax error: {e}",
                severity="error",
                location=f"line {e.lineno}" if hasattr(e, 'lineno') else None
            ))
            return violations
        
        # 3. Import verification
        if self.enable_import_checking:
            import_violations = self._verify_imports(tree)
            violations.extend(import_violations)
        
        # 4. Variable definition checks
        variable_violations = self._verify_variables(tree)
        violations.extend(variable_violations)
        
        # 5. Function structure checks
        function_violations = self._verify_functions(tree, task_requirements)
        violations.extend(function_violations)
        
        # 6. Type checking (if enabled)
        if self.enable_type_checking and self.type_verifier:
            type_violations = self.type_verifier.verify_types(code, tree)
            violations.extend(type_violations)
        
        return violations
    
    def _verify_imports(self, tree: ast.AST) -> List[Violation]:
        """Verify that all imports are valid"""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if not self._is_valid_module(module_name):
                        violations.append(Violation(
                            rule_id="py_import_valid",
                            rule_name="Valid Imports",
                            message=f"Unknown module: {module_name}",
                            severity="error",
                            suggestion=f"Check if {module_name} is installed or misspelled"
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if not self._is_valid_module(module_name):
                        violations.append(Violation(
                            rule_id="py_import_valid",
                            rule_name="Valid Imports",
                            message=f"Unknown module: {module_name}",
                            severity="error"
                        ))
        
        return violations
    
    def _is_valid_module(self, module_name: str) -> bool:
        """Check if a module name is valid"""
        # Check standard library
        if module_name in self.stdlib_modules:
            return True
        
        # Check common third-party libraries  
        if module_name in self.common_libraries:
            return True
        
        # Try to import (simplified check)
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _verify_variables(self, tree: ast.AST) -> List[Violation]:
        """Verify that variables are defined before use"""
        violations = []
        defined_vars = set()
        
        for node in ast.walk(tree):
            # Track variable assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    defined_vars.add(node.target.id)
            elif isinstance(node, ast.FunctionDef):
                defined_vars.add(node.name)
                # Function parameters are also defined
                for arg in node.args.args:
                    defined_vars.add(arg.arg)
            
            # Check variable usage
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined_vars and not self._is_builtin(node.id):
                    violations.append(Violation(
                        rule_id="py_variable_defined",
                        rule_name="Variable Definition",
                        message=f"NameError: '{node.id}' is not defined",
                        severity="error",
                        suggestion=f"Define '{node.id}' before using it"
                    ))
        
        return violations
    
    def _is_builtin(self, name: str) -> bool:
        """Check if name is a Python builtin"""
        builtins = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
            'tuple', 'set', 'bool', 'type', 'isinstance', 'hasattr', 'getattr',
            'sum', 'max', 'min', 'abs', 'round', 'sorted', 'reversed',
            'enumerate', 'zip', 'map', 'filter', 'any', 'all'
        }
        return name in builtins
    
    def _verify_functions(self, tree: ast.AST, requirements: Dict[str, Any]) -> List[Violation]:
        """Verify function structure and requirements"""
        violations = []
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)
                
                # Check if function has return statement when it should
                if self._function_should_return(node):
                    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                    if not has_return:
                        violations.append(Violation(
                            rule_id="py_function_return_type",
                            rule_name="Function Return Type", 
                            message=f"Function '{node.name}' should have a return statement",
                            severity="warning"
                        ))
        
        # Check if required function exists
        required_function = requirements.get("function_name")
        if required_function and required_function not in functions_found:
            violations.append(Violation(
                rule_id="py_function_required",
                rule_name="Required Function",
                message=f"Required function '{required_function}' not found",
                severity="error",
                suggestion=f"Implement function '{required_function}'"
            ))
        
        return violations
    
    def _function_should_return(self, func_node: ast.FunctionDef) -> bool:
        """Determine if function should have return statement"""
        # Simple heuristic: functions with type annotations should return
        if func_node.returns:
            return True
        
        # Functions that don't just print/assign might need to return
        has_non_trivial_ops = False
        for node in ast.walk(func_node):
            if isinstance(node, (ast.BinOp, ast.Compare, ast.Call)):
                has_non_trivial_ops = True
                break
        
        return has_non_trivial_ops
    
    def suggest_initial_solution(self, task_tokens: List[int]) -> Optional[torch.Tensor]:
        """
        Suggest rule-guided initial solution for Python tasks
        
        Args:
            task_tokens: Task specification tokens
            
        Returns:
            Initial solution tokens based on Python templates
        """
        requirements = self._extract_task_requirements(task_tokens)
        
        # Generate basic function template
        function_name = requirements.get("function_name", "solve_task")
        
        template = f"""def {function_name}():
    # TODO: Implement solution
    pass"""
        
        # Convert template to tokens (placeholder)
        # In real implementation, would use actual tokenizer
        template_tokens = self._code_to_tokens(template)
        
        if template_tokens:
            # Add batch dimension [B=1, L]
            return torch.tensor(template_tokens, dtype=torch.long).unsqueeze(0)
        
        return None
    
    def embed_violations(self, violations: List[str]) -> Optional[torch.Tensor]:
        """
        Convert violations to embeddings for neural network

        Args:
            violations: List of violation descriptions

        Returns:
            Violation embeddings [B, V, D] where B=1, V is number of violations
        """
        if not violations:
            return None

        # Create simple embedding based on violation type
        embeddings = []
        for violation in violations:
            # Get violation message (handle both string and Violation objects)
            violation_text = violation.message if hasattr(violation, 'message') else str(violation)

            # Categorize violation type
            if "syntax" in violation_text.lower() or "error" in violation_text.lower():
                # Syntax error embedding
                emb = torch.randn(self.embedding_dim) * 0.1
                emb[0] = 1.0  # Syntax error marker
            elif "import" in violation_text.lower():
                # Import error embedding
                emb = torch.randn(self.embedding_dim) * 0.1
                emb[1] = 1.0  # Import error marker
            elif "name" in violation_text.lower() or "variable" in violation_text.lower():
                # Name/variable error embedding
                emb = torch.randn(self.embedding_dim) * 0.1
                emb[2] = 1.0  # Variable error marker
            elif "type" in violation_text.lower():
                # Type error embedding
                emb = torch.randn(self.embedding_dim) * 0.1
                emb[3] = 1.0  # Type error marker
            else:
                # Generic error embedding
                emb = torch.randn(self.embedding_dim) * 0.1

            embeddings.append(emb)

        # Stack embeddings: [V, D]
        stacked = torch.stack(embeddings)

        # Add batch dimension: [V, D] -> [1, V, D]
        return stacked.unsqueeze(0)
    
    def get_rule_suggestions(self, violations: List[Violation]) -> List[str]:
        """
        Get rule-based suggestions for fixing violations
        
        Args:
            violations: List of detected violations
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        for violation in violations:
            if violation.rule_id == "py_syntax_valid":
                suggestions.append("Check for missing colons, parentheses, or quotes")
            elif violation.rule_id == "py_import_valid":
                suggestions.append(f"Install missing module or check spelling")
            elif violation.rule_id == "py_variable_defined":
                var_name = self._extract_variable_name(violation.message)
                if var_name:
                    suggestions.append(f"Define variable '{var_name}' before using it")
            elif violation.rule_id == "py_function_return_type":
                suggestions.append("Add return statement or return type annotation")
            else:
                suggestions.append("Review code structure and syntax")
        
        return suggestions
    
    def _extract_variable_name(self, message: str) -> Optional[str]:
        """Extract variable name from error message"""
        # Simple regex to extract variable name from NameError message
        match = re.search(r"'([^']+)' is not defined", message)
        return match.group(1) if match else None