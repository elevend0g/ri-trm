#!/usr/bin/env python3
"""
Simple test for RI-TRM core components without external dependencies
"""

import torch
import sys
import os

def test_core_models():
    """Test core model components"""
    print("Testing core model components...")
    
    try:
        # Test if we can import and create basic components
        sys.path.insert(0, '.')
        
        from ri_trm.models.network import TinyRecursiveNetwork
        from ri_trm.models.embedding import InputEmbedding
        
        # Create small network for testing
        network = TinyRecursiveNetwork(
            hidden_size=64,
            num_heads=4, 
            num_layers=2,
            vocab_size=1000
        )
        
        # Test parameter count
        param_count = network.get_parameter_count()
        print(f"✓ Network created with {param_count:,} parameters")
        
        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 10, 64
        x = torch.randn(batch_size, seq_len, hidden_size)
        y = torch.randn(batch_size, seq_len, hidden_size)
        z = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            output = network(x, y, z)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Core models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_basics():
    """Test basic knowledge components"""
    print("\nTesting knowledge components...")
    
    try:
        from ri_trm.knowledge.rule_graph import StructuralRuleGraph, Rule
        from ri_trm.knowledge.path_memory import PathMemoryGraph, ReasoningPath
        
        # Test rule graph
        rule_graph = StructuralRuleGraph(domain="test", embedding_dim=64)
        
        # Add a test rule
        test_rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            rule_type="syntax",
            violation_message="Test violation"
        )
        rule_graph.add_rule(test_rule)
        
        print(f"✓ Rule graph created with {len(rule_graph.rules)} rules")
        
        # Test path memory
        path_memory = PathMemoryGraph(embedding_dim=64, max_paths=100)
        
        # Add a test path
        test_path = ReasoningPath(
            id="test_path",
            error_state="Test error",
            action="Test action", 
            result_state="Test result",
            weight=0.5
        )
        path_memory.add_path(test_path)
        
        stats = path_memory.get_path_statistics()
        print(f"✓ Path memory created with {stats['total_paths']} paths")
        
        return True
        
    except Exception as e:
        print(f"✗ Knowledge components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_domain():
    """Test Python domain setup"""
    print("\nTesting Python domain...")
    
    try:
        from ri_trm.domains.python.rules import PythonRuleVerifier
        from ri_trm.domains.python.verifier import PythonSyntaxVerifier
        
        # Test syntax verifier
        syntax_verifier = PythonSyntaxVerifier()
        
        # Test valid Python code
        valid_code = "def hello():\n    return 'world'"
        violations = syntax_verifier.verify_syntax(valid_code)
        print(f"✓ Syntax verifier works, {len(violations)} violations in valid code")
        
        # Test invalid Python code
        invalid_code = "def hello()  # missing colon\n    return 'world'"
        violations = syntax_verifier.verify_syntax(invalid_code)
        print(f"✓ Syntax verifier detects {len(violations)} violations in invalid code")
        
        # Test rule verifier
        rule_verifier = PythonRuleVerifier(vocab_size=1000, embedding_dim=64)
        print("✓ Python rule verifier created")
        
        return True
        
    except Exception as e:
        print(f"✗ Python domain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dataset():
    """Test basic dataset functionality"""
    print("\nTesting dataset...")
    
    try:
        from ri_trm.training.task_dataset import Task, PythonCodeTaskDataset
        
        # Create a simple task
        task = Task(
            id="test_task",
            description="Test task",
            specification="def test(): pass",
            solution="def test(): return 42",
            tests=[{"input": [], "expected": 42}]
        )
        
        print(f"✓ Task created: {task.id}")
        
        # Create dataset
        dataset = PythonCodeTaskDataset(
            tokenizer_vocab_size=1000,
            max_seq_len=128
        )
        dataset.add_task(task)
        
        print(f"✓ Dataset created with {len(dataset)} tasks")
        
        # Test getting an item
        item = dataset[0]
        print(f"✓ Dataset item shape: {item['description'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all simple tests"""
    print("="*50)
    print("RI-TRM Simple Tests (No External Dependencies)")
    print("="*50)
    
    tests = [
        test_core_models,
        test_knowledge_basics,
        test_python_domain,
        test_simple_dataset
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        print()  # Empty line between tests
    
    print("="*50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All core components working!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)