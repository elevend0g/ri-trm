"""
Python Syntax and Type Verification Components

Specialized verifiers for Python code validation:
- AST-based syntax checking
- Basic type checking integration
"""

import ast
import sys
import tokenize
from io import StringIO
from typing import List, Dict, Set, Optional, Tuple, Any, Union
import re
import tempfile
import subprocess

from ...knowledge.rule_graph import Violation


class PythonSyntaxVerifier:
    """
    AST-based Python syntax verification
    
    Uses Python's built-in AST parser to detect syntax errors
    and structural issues in generated code.
    """
    
    def __init__(self, python_version: Tuple[int, int] = None):
        self.python_version = python_version or sys.version_info[:2]
        
        # Common syntax error patterns
        self.error_patterns = {
            "missing_colon": r"missing ':' at",
            "invalid_syntax": r"invalid syntax",
            "indentation": r"indentation",
            "unmatched_paren": r"unmatched",
            "eof_error": r"unexpected EOF"
        }
    
    def verify_syntax(self, code: str) -> List[Violation]:
        """
        Verify Python syntax using AST parser
        
        Args:
            code: Python code string to verify
            
        Returns:
            List of syntax violations
        """
        violations = []
        
        if not code.strip():
            violations.append(Violation(
                rule_id="py_empty_code",
                rule_name="Non-empty Code",
                message="Code cannot be empty",
                severity="error"
            ))
            return violations
        
        # 1. Check basic tokenization
        tokenize_violations = self._check_tokenization(code)
        violations.extend(tokenize_violations)
        
        # 2. Check AST parsing
        ast_violations = self._check_ast_parsing(code)
        violations.extend(ast_violations)
        
        # 3. Check indentation consistency
        if not ast_violations:  # Only if AST parsing succeeded
            indent_violations = self._check_indentation(code)
            violations.extend(indent_violations)
        
        return violations
    
    def _check_tokenization(self, code: str) -> List[Violation]:
        """Check if code can be properly tokenized"""
        violations = []
        
        try:
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))
        except tokenize.TokenError as e:
            violations.append(Violation(
                rule_id="py_tokenize_error",
                rule_name="Tokenization Error",
                message=f"Tokenization failed: {e}",
                severity="error"
            ))
        except IndentationError as e:
            violations.append(Violation(
                rule_id="py_indentation_error",
                rule_name="Indentation Error",
                message=f"Indentation error: {e}",
                severity="error",
                location=f"line {e.lineno}" if hasattr(e, 'lineno') else None
            ))
        
        return violations
    
    def _check_ast_parsing(self, code: str) -> List[Violation]:
        """Check if code can be parsed into AST"""
        violations = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            # Categorize the syntax error
            error_type = self._categorize_syntax_error(str(e))
            
            violations.append(Violation(
                rule_id=f"py_syntax_{error_type}",
                rule_name=f"Syntax Error ({error_type})",
                message=f"Syntax error: {e.msg}",
                severity="error",
                location=f"line {e.lineno}, col {e.offset}" if e.lineno else None,
                suggestion=self._get_syntax_suggestion(error_type, str(e))
            ))
        except Exception as e:
            violations.append(Violation(
                rule_id="py_parse_error",
                rule_name="Parse Error",
                message=f"Failed to parse code: {e}",
                severity="error"
            ))
        
        return violations
    
    def _categorize_syntax_error(self, error_msg: str) -> str:
        """Categorize syntax error type"""
        error_msg_lower = error_msg.lower()
        
        for pattern_name, pattern in self.error_patterns.items():
            if re.search(pattern, error_msg_lower):
                return pattern_name
        
        return "unknown"
    
    def _get_syntax_suggestion(self, error_type: str, error_msg: str) -> str:
        """Get suggestion for fixing syntax error"""
        suggestions = {
            "missing_colon": "Add ':' at the end of if/for/while/def statements",
            "invalid_syntax": "Check for typos, missing operators, or invalid keywords",
            "indentation": "Use consistent indentation (4 spaces recommended)",
            "unmatched_paren": "Check for matching parentheses, brackets, or braces",
            "eof_error": "Check for incomplete statements or missing closing symbols"
        }
        
        return suggestions.get(error_type, "Review code syntax and structure")
    
    def _check_indentation(self, code: str) -> List[Violation]:
        """Check for consistent indentation"""
        violations = []
        lines = code.split('\n')
        
        indent_levels = []
        uses_tabs = False
        uses_spaces = False
        
        for i, line in enumerate(lines, 1):
            if line.strip():  # Skip empty lines
                # Count leading whitespace
                leading_ws = len(line) - len(line.lstrip())
                if leading_ws > 0:
                    if '\t' in line[:leading_ws]:
                        uses_tabs = True
                    if ' ' in line[:leading_ws]:
                        uses_spaces = True
                    
                    indent_levels.append((i, leading_ws, line[:leading_ws]))
        
        # Check for mixed tabs and spaces
        if uses_tabs and uses_spaces:
            violations.append(Violation(
                rule_id="py_mixed_indentation",
                rule_name="Mixed Indentation",
                message="Mixing tabs and spaces for indentation",
                severity="error",
                suggestion="Use either tabs or spaces consistently (spaces recommended)"
            ))
        
        # Check for irregular indentation levels (when using spaces)
        if uses_spaces and not uses_tabs:
            space_counts = [level for _, level, ws in indent_levels if ' ' in ws]
            if space_counts:
                # Check if all indentation is multiple of 4
                irregular = [count for count in space_counts if count % 4 != 0]
                if irregular:
                    violations.append(Violation(
                        rule_id="py_irregular_indentation",
                        rule_name="Irregular Indentation",
                        message="Indentation should be multiple of 4 spaces",
                        severity="warning",
                        suggestion="Use 4 spaces for each indentation level"
                    ))
        
        return violations


class PythonTypeVerifier:
    """
    Basic Python type checking integration
    
    Provides simple type checking capabilities using mypy integration
    or basic AST analysis for type annotations.
    """
    
    def __init__(self, use_mypy: bool = False):
        self.use_mypy = use_mypy
        self.mypy_available = False
        
        if use_mypy:
            try:
                import mypy.api
                self.mypy_available = True
            except ImportError:
                self.mypy_available = False
    
    def verify_types(self, code: str, tree: ast.AST = None) -> List[Violation]:
        """
        Verify type annotations and basic type consistency
        
        Args:
            code: Python code string
            tree: Parsed AST (optional, will parse if not provided)
            
        Returns:
            List of type-related violations
        """
        violations = []
        
        if tree is None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                # Can't check types if syntax is invalid
                return violations
        
        # 1. Check type annotation consistency
        annotation_violations = self._check_type_annotations(tree)
        violations.extend(annotation_violations)
        
        # 2. Use mypy if available
        if self.mypy_available and self.use_mypy:
            mypy_violations = self._run_mypy_check(code)
            violations.extend(mypy_violations)
        
        # 3. Basic type inference checks
        inference_violations = self._check_basic_type_inference(tree)
        violations.extend(inference_violations)
        
        return violations
    
    def _check_type_annotations(self, tree: ast.AST) -> List[Violation]:
        """Check for consistent and proper type annotations"""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has return annotation when it should
                has_return_stmt = any(isinstance(n, ast.Return) and n.value is not None 
                                    for n in ast.walk(node))
                
                if has_return_stmt and node.returns is None:
                    violations.append(Violation(
                        rule_id="py_missing_return_annotation",
                        rule_name="Missing Return Annotation",
                        message=f"Function '{node.name}' should have return type annotation",
                        severity="warning",
                        suggestion="Add return type annotation: -> ReturnType"
                    ))
                
                # Check parameter annotations
                unannotated_params = [arg.arg for arg in node.args.args if arg.annotation is None]
                if unannotated_params and len(node.args.args) > 0:
                    violations.append(Violation(
                        rule_id="py_missing_param_annotation",
                        rule_name="Missing Parameter Annotation",
                        message=f"Parameters {unannotated_params} missing type annotations",
                        severity="info",
                        suggestion="Add type annotations to function parameters"
                    ))
        
        return violations
    
    def _run_mypy_check(self, code: str) -> List[Violation]:
        """Run mypy type checking on code"""
        violations = []
        
        if not self.mypy_available:
            return violations
        
        try:
            import mypy.api
            
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run mypy
            stdout, stderr, exit_code = mypy.api.run([
                '--ignore-missing-imports',
                '--no-error-summary',
                temp_file
            ])
            
            # Parse mypy output
            if stdout:
                mypy_violations = self._parse_mypy_output(stdout)
                violations.extend(mypy_violations)
            
            # Clean up
            import os
            os.unlink(temp_file)
            
        except Exception as e:
            # Mypy check failed, but don't fail the whole verification
            pass
        
        return violations
    
    def _parse_mypy_output(self, mypy_output: str) -> List[Violation]:
        """Parse mypy output into violations"""
        violations = []
        
        for line in mypy_output.strip().split('\n'):
            if ':' in line and 'error:' in line:
                # Parse mypy error format: file.py:line: error: message
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    line_no = parts[1]
                    message = parts[3].strip()
                    
                    violations.append(Violation(
                        rule_id="py_mypy_error",
                        rule_name="Type Error",
                        message=f"Type error: {message}",
                        severity="error",
                        location=f"line {line_no}"
                    ))
        
        return violations
    
    def _check_basic_type_inference(self, tree: ast.AST) -> List[Violation]:
        """Basic type consistency checks without full inference"""
        violations = []
        
        # Check for obvious type mismatches
        for node in ast.walk(tree):
            # Check string/number operations
            if isinstance(node, ast.BinOp):
                violations.extend(self._check_binop_types(node))
            
            # Check function call argument counts
            elif isinstance(node, ast.Call):
                violations.extend(self._check_call_args(node))
        
        return violations
    
    def _check_binop_types(self, node: ast.BinOp) -> List[Violation]:
        """Check binary operation type compatibility"""
        violations = []
        
        # Simple checks for obvious mismatches
        left_type = self._infer_simple_type(node.left)
        right_type = self._infer_simple_type(node.right)
        
        if left_type and right_type:
            # Check string + number (common error)
            if isinstance(node.op, ast.Add):
                if (left_type == 'str' and right_type in ['int', 'float']) or \
                   (right_type == 'str' and left_type in ['int', 'float']):
                    violations.append(Violation(
                        rule_id="py_string_number_add",
                        rule_name="String Number Addition",
                        message="Cannot add string and number",
                        severity="error",
                        suggestion="Convert number to string or use string formatting"
                    ))
        
        return violations
    
    def _infer_simple_type(self, node: ast.AST) -> Optional[str]:
        """Simple type inference for basic literals"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return 'str'
            elif isinstance(node.value, int):
                return 'int'
            elif isinstance(node.value, float):
                return 'float'
            elif isinstance(node.value, bool):
                return 'bool'
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Dict):
            return 'dict'
        
        return None
    
    def _check_call_args(self, node: ast.Call) -> List[Violation]:
        """Check function call argument patterns"""
        violations = []
        
        # Check built-in function argument counts
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Simple checks for common built-ins
            expected_args = {
                'len': 1,
                'abs': 1,
                'int': (1, 2),  # int(x) or int(x, base)
                'str': (0, 3),  # str() or str(x) or str(x, encoding, errors)
                'range': (1, 3),  # range(stop) or range(start, stop) or range(start, stop, step)
            }
            
            if func_name in expected_args:
                expected = expected_args[func_name]
                actual = len(node.args)
                
                if isinstance(expected, int):
                    if actual != expected:
                        violations.append(Violation(
                            rule_id="py_wrong_arg_count",
                            rule_name="Wrong Argument Count",
                            message=f"{func_name}() takes {expected} arguments but {actual} given",
                            severity="error"
                        ))
                elif isinstance(expected, tuple):
                    min_args, max_args = expected
                    if not (min_args <= actual <= max_args):
                        violations.append(Violation(
                            rule_id="py_wrong_arg_count",
                            rule_name="Wrong Argument Count",
                            message=f"{func_name}() takes {min_args}-{max_args} arguments but {actual} given",
                            severity="error"
                        ))
        
        return violations