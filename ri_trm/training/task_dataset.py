"""
Task-Based Dataset for RI-TRM

Unlike traditional token-based training, RI-TRM trains on complete tasks:
problem-solution pairs with test cases for verification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import random
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Task:
    """A single task for training"""
    id: str
    description: str
    specification: str
    solution: Optional[str] = None
    tests: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


@dataclass
class TaskBatch:
    """Batch of tasks for training"""
    task_ids: List[str]
    descriptions: torch.Tensor  # [B, L]  
    specifications: torch.Tensor  # [B, L]
    solutions: Optional[torch.Tensor] = None  # [B, L]
    tests: Optional[List[List[Dict]]] = None
    metadata: Optional[List[Dict]] = None


class TaskDataset(Dataset, ABC):
    """
    Abstract base class for task-based datasets
    
    Implements the core interface for task-based training where
    each example is a complete problem-solution pair rather than
    individual tokens.
    """
    
    def __init__(
        self,
        tokenizer_vocab_size: int = 32000,
        max_seq_len: int = 512,
        include_solutions: bool = True
    ):
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.max_seq_len = max_seq_len
        self.include_solutions = include_solutions
        
        # Placeholder tokenizer (would use real tokenizer in practice)
        self.token_to_text = {}
        self.text_to_token = {}
        self.next_token_id = 1  # 0 reserved for padding
        
        # Special tokens
        self.pad_token = 0
        self.sos_token = self._get_token_id("<SOS>")
        self.eos_token = self._get_token_id("<EOS>")
        self.unk_token = self._get_token_id("<UNK>")
        
        # Tasks storage
        self.tasks: List[Task] = []
    
    def _get_token_id(self, text: str) -> int:
        """Get or create token ID for text"""
        if text not in self.text_to_token:
            token_id = self.next_token_id
            self.text_to_token[text] = token_id
            self.token_to_text[token_id] = text
            self.next_token_id += 1
        return self.text_to_token[text]
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Simple tokenization (replace with real tokenizer)"""
        # Basic word-level tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = [self.sos_token]
        
        for word in words:
            tokens.append(self._get_token_id(word))
        
        tokens.append(self.eos_token)
        return tokens
    
    def _pad_sequence(self, tokens: List[int]) -> torch.Tensor:
        """Pad token sequence to max length"""
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len-1] + [self.eos_token]
        else:
            tokens = tokens + [self.pad_token] * (self.max_seq_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    @abstractmethod
    def load_tasks(self, data_source: str):
        """Load tasks from data source"""
        pass
    
    @abstractmethod
    def generate_synthetic_tasks(self, num_tasks: int) -> List[Task]:
        """Generate synthetic tasks for training"""
        pass
    
    def add_task(self, task: Task):
        """Add a task to the dataset"""
        self.tasks.append(task)
    
    def add_tasks(self, tasks: List[Task]):
        """Add multiple tasks to the dataset"""
        self.tasks.extend(tasks)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single task"""
        task = self.tasks[idx]
        
        # Tokenize description and specification
        desc_tokens = self._tokenize_text(task.description)
        spec_tokens = self._tokenize_text(task.specification)
        
        # Pad sequences
        desc_tensor = self._pad_sequence(desc_tokens)
        spec_tensor = self._pad_sequence(spec_tokens)
        
        item = {
            "task_id": task.id,
            "description": desc_tensor,
            "specification": spec_tensor,
            "tests": task.tests or [],
            "metadata": task.metadata or {}
        }
        
        # Include solution if available
        if self.include_solutions and task.solution:
            sol_tokens = self._tokenize_text(task.solution)
            sol_tensor = self._pad_sequence(sol_tokens)
            item["solution"] = sol_tensor
        
        return item
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> TaskBatch:
        """Collate function for DataLoader"""
        batch_size = len(batch)
        
        # Stack tensors
        descriptions = torch.stack([item["description"] for item in batch])
        specifications = torch.stack([item["specification"] for item in batch])
        
        solutions = None
        if "solution" in batch[0]:
            solutions = torch.stack([item["solution"] for item in batch])
        
        return TaskBatch(
            task_ids=[item["task_id"] for item in batch],
            descriptions=descriptions,
            specifications=specifications,
            solutions=solutions,
            tests=[item["tests"] for item in batch],
            metadata=[item["metadata"] for item in batch]
        )
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Create DataLoader for the dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


class PythonCodeTaskDataset(TaskDataset):
    """
    Dataset for Python code generation tasks
    
    Focuses on programming problems with executable test cases,
    ideal for demonstrating RI-TRM's rule-based verification.
    """
    
    def __init__(
        self,
        tokenizer_vocab_size: int = 32000,
        max_seq_len: int = 512,
        include_solutions: bool = True,
        difficulty_levels: List[str] = None,
        tokenizer_name: str = "gpt2"
    ):
        super().__init__(tokenizer_vocab_size, max_seq_len, include_solutions)

        self.difficulty_levels = difficulty_levels or ["easy", "medium", "hard"]

        # Real tokenizer integration
        from ..tokenizer import get_tokenizer
        self.real_tokenizer = get_tokenizer(model_name=tokenizer_name, max_length=max_seq_len)

        # Update vocab size and special tokens from real tokenizer
        self.tokenizer_vocab_size = self.real_tokenizer.vocab_size
        self.pad_token = self.real_tokenizer.pad_token_id
        self.eos_token = self.real_tokenizer.eos_token_id

    def _tokenize_text(self, text: str) -> List[int]:
        """Use real tokenizer instead of placeholder"""
        return self.real_tokenizer.encode(text, padding=False, return_tensors=False)

    def _pad_sequence(self, tokens: List[int]) -> torch.Tensor:
        """Pad using real tokenizer's padding token"""
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        padded = tokens + [self.pad_token] * (self.max_seq_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def load_tasks(self, data_source: str):
        """Load Python coding tasks from JSON file"""
        try:
            with open(data_source, 'r') as f:
                data = json.load(f)
            
            for task_data in data.get("tasks", []):
                task = Task(
                    id=task_data["id"],
                    description=task_data["description"],
                    specification=task_data["specification"],
                    solution=task_data.get("solution"),
                    tests=task_data.get("tests", []),
                    metadata=task_data.get("metadata", {})
                )
                self.add_task(task)
                
        except Exception as e:
            print(f"Warning: Could not load tasks from {data_source}: {e}")
    
    def generate_synthetic_tasks(self, num_tasks: int) -> List[Task]:
        """Generate synthetic Python coding tasks"""
        synthetic_tasks = []
        
        # Task templates for different difficulty levels
        templates = {
            "easy": [
                {
                    "name": "simple_arithmetic",
                    "description": "Write a function that performs basic arithmetic",
                    "template": "Write a function `{func_name}` that takes two numbers and returns their {operation}."
                },
                {
                    "name": "string_manipulation",
                    "description": "Write a function that manipulates strings",
                    "template": "Write a function `{func_name}` that takes a string and returns {transformation}."
                },
                {
                    "name": "list_operations",
                    "description": "Write a function that works with lists",
                    "template": "Write a function `{func_name}` that takes a list and returns {list_operation}."
                }
            ],
            "medium": [
                {
                    "name": "data_processing",
                    "description": "Write a function that processes data structures",
                    "template": "Write a function `{func_name}` that processes {data_type} and {processing_task}."
                },
                {
                    "name": "algorithm_implementation",
                    "description": "Implement a standard algorithm",
                    "template": "Implement the {algorithm_name} algorithm in a function called `{func_name}`."
                }
            ],
            "hard": [
                {
                    "name": "complex_algorithm",
                    "description": "Implement a complex algorithmic solution",
                    "template": "Solve the {problem_name} problem by implementing `{func_name}` with {constraints}."
                }
            ]
        }
        
        # Generate tasks
        for i in range(num_tasks):
            difficulty = random.choice(self.difficulty_levels)
            template = random.choice(templates[difficulty])
            
            task = self._generate_task_from_template(f"synthetic_{i:04d}", template, difficulty)
            synthetic_tasks.append(task)
        
        return synthetic_tasks
    
    def _generate_task_from_template(self, task_id: str, template: Dict, difficulty: str) -> Task:
        """Generate a specific task from template"""
        
        # Generate parameters based on template
        if template["name"] == "simple_arithmetic":
            operations = ["sum", "difference", "product", "quotient"]
            operation = random.choice(operations)
            func_name = f"calculate_{operation}"
            
            description = template["template"].format(
                func_name=func_name,
                operation=operation
            )
            
            # Generate solution
            op_map = {
                "sum": "+", "difference": "-", 
                "product": "*", "quotient": "/"
            }
            solution = f"""def {func_name}(a, b):
    return a {op_map[operation]} b"""
            
            # Generate tests
            tests = [
                {"input": [2, 3], "expected": eval(f"2 {op_map[operation]} 3")},
                {"input": [10, 5], "expected": eval(f"10 {op_map[operation]} 5")},
                {"input": [0, 1], "expected": eval(f"0 {op_map[operation]} 1")}
            ]
            
        elif template["name"] == "string_manipulation":
            transformations = [
                "it in uppercase", "it in lowercase", 
                "it reversed", "its length"
            ]
            transformation = random.choice(transformations)
            func_name = "transform_string"
            
            description = template["template"].format(
                func_name=func_name,
                transformation=transformation
            )
            
            # Generate solution based on transformation
            if "uppercase" in transformation:
                solution = f"""def {func_name}(s):
    return s.upper()"""
                tests = [
                    {"input": ["hello"], "expected": "HELLO"},
                    {"input": ["World"], "expected": "WORLD"}
                ]
            elif "lowercase" in transformation:
                solution = f"""def {func_name}(s):
    return s.lower()"""
                tests = [
                    {"input": ["HELLO"], "expected": "hello"},
                    {"input": ["World"], "expected": "world"}
                ]
            elif "reversed" in transformation:
                solution = f"""def {func_name}(s):
    return s[::-1]"""
                tests = [
                    {"input": ["hello"], "expected": "olleh"},
                    {"input": ["abc"], "expected": "cba"}
                ]
            else:  # length
                solution = f"""def {func_name}(s):
    return len(s)"""
                tests = [
                    {"input": ["hello"], "expected": 5},
                    {"input": [""], "expected": 0}
                ]
                
        else:
            # Default simple task
            func_name = "solve_task"
            description = "Write a function that solves the given task."
            solution = f"""def {func_name}():
    pass"""
            tests = []
        
        return Task(
            id=task_id,
            description=description,
            specification=description,  # Same for synthetic tasks
            solution=solution,
            tests=tests,
            metadata={
                "difficulty": difficulty,
                "template": template["name"],
                "synthetic": True
            }
        )
    
    def create_humaneval_subset(self, num_problems: int = 20) -> List[Task]:
        """
        Create a subset of HumanEval-style problems
        
        This would integrate with the actual HumanEval dataset
        in a real implementation.
        """
        humaneval_style_tasks = []
        
        # Simplified HumanEval-style problems
        problems = [
            {
                "id": "humaneval_001",
                "description": "Check if a number is prime",
                "specification": """def is_prime(n):
    \"\"\"
    Return True if n is a prime number, False otherwise.
    
    >>> is_prime(2)
    True
    >>> is_prime(4)
    False
    >>> is_prime(17)
    True
    \"\"\"
    pass""",
                "solution": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
                "tests": [
                    {"input": [2], "expected": True},
                    {"input": [4], "expected": False},
                    {"input": [17], "expected": True},
                    {"input": [1], "expected": False}
                ]
            },
            {
                "id": "humaneval_002", 
                "description": "Calculate factorial",
                "specification": """def factorial(n):
    \"\"\"
    Calculate the factorial of n.
    
    >>> factorial(5)
    120
    >>> factorial(0)
    1
    \"\"\"
    pass""",
                "solution": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
                "tests": [
                    {"input": [5], "expected": 120},
                    {"input": [0], "expected": 1},
                    {"input": [3], "expected": 6}
                ]
            },
            {
                "id": "humaneval_003",
                "description": "Find maximum in list",
                "specification": """def find_max(numbers):
    \"\"\"
    Find the maximum number in a list.
    
    >>> find_max([1, 3, 2, 5, 4])
    5
    >>> find_max([-1, -3, -2])
    -1
    \"\"\"
    pass""",
                "solution": """def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val""",
                "tests": [
                    {"input": [[1, 3, 2, 5, 4]], "expected": 5},
                    {"input": [[-1, -3, -2]], "expected": -1},
                    {"input": [[42]], "expected": 42}
                ]
            }
        ]
        
        # Convert to Task objects
        for prob in problems[:num_problems]:
            task = Task(
                id=prob["id"],
                description=prob["description"],
                specification=prob["specification"],
                solution=prob["solution"],
                tests=prob["tests"],
                metadata={"source": "humaneval", "difficulty": "medium"}
            )
            humaneval_style_tasks.append(task)
        
        return humaneval_style_tasks
    
    def validate_task_solution(self, task: Task) -> Dict[str, Any]:
        """
        Validate that the task solution passes its tests
        
        Args:
            task: Task to validate
            
        Returns:
            Validation results
        """
        if not task.solution or not task.tests:
            return {"valid": False, "reason": "missing_solution_or_tests"}
        
        try:
            # Execute solution (simplified - would use proper sandboxing)
            exec_globals = {}
            exec(task.solution, exec_globals)
            
            # Find the function (assume single function for now)
            func_name = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    func_name = name
                    break
            
            if not func_name:
                return {"valid": False, "reason": "no_function_found"}
            
            func = exec_globals[func_name]
            
            # Run tests
            passed_tests = 0
            failed_tests = []
            
            for i, test in enumerate(task.tests):
                try:
                    if isinstance(test["input"], list):
                        result = func(*test["input"])
                    else:
                        result = func(test["input"])
                    
                    if result == test["expected"]:
                        passed_tests += 1
                    else:
                        failed_tests.append({
                            "test_id": i,
                            "input": test["input"],
                            "expected": test["expected"],
                            "actual": result
                        })
                        
                except Exception as e:
                    failed_tests.append({
                        "test_id": i,
                        "input": test["input"],
                        "error": str(e)
                    })
            
            return {
                "valid": len(failed_tests) == 0,
                "passed_tests": passed_tests,
                "total_tests": len(task.tests),
                "failed_tests": failed_tests
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"execution_error: {e}"}
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        if not self.tasks:
            return {"total_tasks": 0}
        
        # Count by difficulty
        difficulty_counts = {}
        for task in self.tasks:
            difficulty = task.metadata.get("difficulty", "unknown")
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Count by source
        source_counts = {}
        for task in self.tasks:
            source = task.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Average sequence lengths
        desc_lengths = []
        spec_lengths = []
        sol_lengths = []
        
        for task in self.tasks:
            desc_lengths.append(len(self._tokenize_text(task.description)))
            spec_lengths.append(len(self._tokenize_text(task.specification)))
            if task.solution:
                sol_lengths.append(len(self._tokenize_text(task.solution)))
        
        return {
            "total_tasks": len(self.tasks),
            "difficulty_distribution": difficulty_counts,
            "source_distribution": source_counts,
            "average_description_length": sum(desc_lengths) / len(desc_lengths),
            "average_specification_length": sum(spec_lengths) / len(spec_lengths),
            "average_solution_length": sum(sol_lengths) / len(sol_lengths) if sol_lengths else 0,
            "vocabulary_size": len(self.text_to_token)
        }