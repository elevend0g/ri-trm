#!/usr/bin/env python3
"""
Basic smoke test for RI-TRM implementation
Tests core functionality without full training
"""

import torch
import sys
import os

# Add ri_trm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import all modules globally
from ri_trm.models.network import TinyRecursiveNetwork
from ri_trm.models.embedding import InputEmbedding, OutputEmbedding, LatentEmbedding
from ri_trm.models.heads import OutputHead, ConfidenceHead
from ri_trm.knowledge.rule_graph import StructuralRuleGraph
from ri_trm.knowledge.fact_graph import FactualKnowledgeGraph
from ri_trm.knowledge.path_memory import PathMemoryGraph
from ri_trm.domains.python.setup import PythonDomainSetup
from ri_trm.inference.recursive_solver import RecursiveRefinementSolver
from ri_trm.training.trainer import RITRMTrainer, TrainingConfig
from ri_trm.training.task_dataset import PythonCodeTaskDataset
from ri_trm.evaluation.benchmarks import HumanEvalBenchmark

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Imports are already done globally, just test they work
        assert TinyRecursiveNetwork is not None
        assert InputEmbedding is not None
        assert PythonDomainSetup is not None
        print("✓ All imports successful")
        return True
    except (ImportError, NameError) as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test creating model components"""
    print("\nTesting model creation...")
    
    try:
        # Test network creation
        network = TinyRecursiveNetwork(
            hidden_size=64,  # Smaller for testing
            num_heads=4,
            num_layers=2,
            vocab_size=1000
        )
        
        # Test embeddings
        input_emb = InputEmbedding(vocab_size=1000, hidden_size=64, max_seq_len=128)
        output_emb = OutputEmbedding(vocab_size=1000, hidden_size=64, max_seq_len=128)
        latent_emb = LatentEmbedding(hidden_size=64, max_seq_len=128)
        
        # Test heads
        output_head = OutputHead(hidden_size=64, vocab_size=1000)
        confidence_head = ConfidenceHead(hidden_size=64)
        
        param_count = network.get_parameter_count()
        print(f"✓ Model created with {param_count:,} parameters")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_knowledge_components():
    """Test knowledge architecture setup"""
    print("\nTesting knowledge components...")
    
    try:
        # Test domain setup
        domain_setup = PythonDomainSetup(
            embedding_dim=64,
            vocab_size=1000
        )
        
        components = domain_setup.setup_complete_domain()
        
        # Check components exist
        assert "rule_graph" in components
        assert "fact_graph" in components  
        assert "path_memory" in components
        
        # Test basic functionality
        rule_graph = components["rule_graph"]
        path_memory = components["path_memory"]
        
        # Test rule coverage
        coverage = rule_graph.get_rule_coverage()
        print(f"✓ Rule graph: {sum(coverage.values())} total rules")
        
        # Test path memory
        stats = path_memory.get_path_statistics()
        print(f"✓ Path memory: {stats['total_paths']} initial paths")
        
        return True
        
    except Exception as e:
        print(f"✗ Knowledge components failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset and task creation"""
    print("\nTesting dataset creation...")
    
    try:
        # Create dataset
        dataset = PythonCodeTaskDataset(
            tokenizer_vocab_size=1000,
            max_seq_len=128,
            include_solutions=True
        )
        
        # Generate synthetic tasks
        tasks = dataset.generate_synthetic_tasks(5)
        dataset.add_tasks(tasks)
        
        print(f"✓ Created dataset with {len(dataset)} tasks")
        
        # Test dataloader
        dataloader = dataset.create_dataloader(batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"✓ Dataloader created, batch shape: {batch.descriptions.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass through complete system"""
    print("\nTesting forward pass...")
    
    try:
        # Create minimal setup
        domain_setup = PythonDomainSetup(embedding_dim=64, vocab_size=1000)
        components = domain_setup.setup_complete_domain()
        
        # Create model
        network = TinyRecursiveNetwork(hidden_size=64, num_layers=2, vocab_size=1000)
        input_emb = InputEmbedding(vocab_size=1000, hidden_size=64, max_seq_len=128)
        output_emb = OutputEmbedding(vocab_size=1000, hidden_size=64, max_seq_len=128)
        latent_emb = LatentEmbedding(hidden_size=64, max_seq_len=128)
        output_head = OutputHead(hidden_size=64, vocab_size=1000)
        confidence_head = ConfidenceHead(hidden_size=64)
        
        # Create solver
        solver = RecursiveRefinementSolver(
            network=network,
            input_embedding=input_emb,
            output_embedding=output_emb,
            latent_embedding=latent_emb,
            output_head=output_head,
            confidence_head=confidence_head,
            vocab_size=1000,
            max_iterations=3,  # Small for testing
            reasoning_steps=2
        )
        
        # Connect knowledge
        solver.set_knowledge_components(
            rule_verifier=components["rule_graph"].verifier,
            path_memory=components["path_memory"],
            factual_kg=components["fact_graph"]
        )
        
        # Test forward pass
        test_input = torch.randint(1, 1000, (1, 10))  # Random tokens
        
        with torch.no_grad():
            result = solver(test_input, return_trace=True, early_stopping=True)
        
        print(f"✓ Forward pass successful")
        print(f"  - Result type: {type(result)}")
        print(f"  - Solution shape: {result.solution.shape}")
        print(f"  - Reasoning steps: {result.num_steps}")
        print(f"  - Final confidence: {result.final_confidence:.3f}")
        print(f"  - Converged: {result.converged}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all basic tests"""
    print("="*50)
    print("RI-TRM Basic Functionality Tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_model_creation, 
        test_knowledge_components,
        test_dataset_creation,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RI-TRM implementation is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)