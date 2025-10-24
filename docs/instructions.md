## Immediate Action Plan

### **Priority 1: Tokenization**

```python
# Replace placeholder tokenization with real tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Update embedding.py:
class InputEmbedding:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.embed = nn.Embedding(
            len(self.tokenizer), 
            hidden_dim
        )
    
    def forward(self, text):
        tokens = self.tokenizer.encode(text)
        return self.embed(tokens)
```

**Why critical:** Without this, your generated code will be gibberish.

### **Priority 2: Expand K_R**

Add these rules to `domains/python/rules.py`:

```python
SYNTAX_RULES = {
    'colon_after_def': r'def\s+\w+\([^)]*\)\s*:',
    'colon_after_if': r'if\s+.*:\s*$',
    'colon_after_for': r'for\s+.*:\s*$', 
    'colon_after_while': r'while\s+.*:\s*$',
    'colon_after_class': r'class\s+\w+.*:\s*$',
    'indentation_4spaces': r'^( {4})*[^ ]',  # Must use 4-space indents
    'matched_parens': 'count("(") == count(")")',
    'matched_brackets': 'count("[") == count("]")',
    'matched_braces': 'count("{") == count("}")',
    'no_tabs': 'not contains("\t")',  # Only spaces, no tabs
}

TYPE_RULES = {
    'return_annotation': 'def with return must have -> type',
    'param_annotation': 'function params should have types',
    'no_bare_except': 'except: must specify exception type',
}

# Use AST for verification:
import ast

def verify_code(code_str):
    violations = []
    
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} at line {e.lineno}"]
    
    # Check for common issues
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.returns is None and has_return_statement(node):
                violations.append(f"Function {node.name} missing return type annotation")
        
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                violations.append(f"Bare except at line {node.lineno}")
    
    return violations
```

### **Priority 3: TRM Baseline**

Create `models/trm_baseline.py`:

```python
class TRMBaseline(nn.Module):
    """Pure TRM without K_R - for ablation"""
    
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()
        self.use_rules = False  # KEY DIFFERENCE
        
        # Same architecture as RI-TRM
        self.embedding = InputEmbedding(vocab_size, hidden_dim)
        self.network = TinyNetwork(hidden_dim, layers=2)
        self.output_head = OutputHead(hidden_dim, vocab_size)
        
    def recursive_refinement(self, task, max_steps=16):
        # Same as RI-TRM but WITHOUT rule verification
        solution = self.initial_solution(task)
        
        for step in range(max_steps):
            # NO RULE CHECK - pure learning
            solution = self.improve_solution(solution)
            
            if self.is_confident(solution):
                break
                
        return solution
```

Then compare:

```python
# Critical experiment:
ri_trm = RITRMModel(use_rules=True)  
trm_baseline = TRMBaseline(use_rules=False)

ri_trm_acc = train_and_evaluate(ri_trm, tasks)
baseline_acc = train_and_evaluate(trm_baseline, tasks)

print(f"RI-TRM: {ri_trm_acc:.1%}")
print(f"TRM Baseline: {baseline_acc:.1%}")
print(f"Gain from rules: {ri_trm_acc - baseline_acc:.1%}")
```
