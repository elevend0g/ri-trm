# Contributing to RI-TRM

Thank you for your interest in contributing to the Rule-Initialized Tiny Recursive Model (RI-TRM) project! This document provides guidelines for contributing to this research implementation.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ri-trm.git
   cd ri-trm
   ```
3. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the demo** to ensure everything works:
   ```bash
   python demo.py
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where possible
- Include docstrings for all functions and classes
- Keep functions focused and modular

### Project Structure

When adding new features, follow the existing structure:

```
ri_trm/
├── models/          # Neural network components
├── knowledge/       # Knowledge graph components  
├── domains/         # Domain-specific implementations
├── inference/       # Reasoning and generation
├── training/        # Training pipelines
└── evaluation/      # Benchmarks and metrics
```

### Testing

- Add tests for new functionality
- Ensure existing tests still pass
- Test edge cases and error conditions

## Types of Contributions

### 1. Bug Fixes
- Report bugs via GitHub issues
- Include minimal reproduction case
- Submit fixes with clear commit messages

### 2. New Features

#### High Priority Areas:
- **Real Tokenization**: Replace placeholder tokenization with actual tokenizers (GPT, T5, etc.)
- **Extended Domains**: Add support for mathematics, SQL, formal verification
- **Advanced Rule Systems**: More sophisticated rule graphs and verification
- **Evaluation**: More comprehensive benchmarks and human evaluation

#### Medium Priority:
- **Transfer Learning**: Cross-domain path memory sharing
- **Optimization**: Training speed and memory efficiency improvements
- **Visualization**: Tools for inspecting reasoning traces and path memory

#### Research Extensions:
- **Hierarchical Path Composition**: Combining atomic paths into complex strategies
- **Multi-Agent Systems**: Collaborative RI-TRM instances
- **Automatic Rule Discovery**: Learning rules from examples

### 3. Documentation
- Improve README and documentation
- Add tutorials and examples
- Create architecture diagrams

### 4. Performance Improvements
- Training efficiency optimizations
- Memory usage reductions
- Inference speed improvements

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test thoroughly**:
   ```bash
   python demo.py  # Ensure demo still works
   # Run any additional tests
   ```

4. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes
   
   - Detailed point 1
   - Detailed point 2
   - Any breaking changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots/examples if applicable

### Commit Message Format

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `perf:` Performance improvements

## Research Contributions

### Academic Extensions

If you're extending RI-TRM for research:

1. **Document your approach** in detail
2. **Include experimental results** and comparisons
3. **Provide reproducible experiments**
4. **Consider submitting to workshops/conferences**

### Citing This Work

If you use or extend RI-TRM in your research, please cite:

```bibtex
@software{ri_trm_2024,
  title={Rule-Initialized Tiny Recursive Model (RI-TRM)},
  author={RI-TRM Research Team},
  year={2024},
  url={https://github.com/your-username/ri-trm},
  version={0.1.0}
}
```

## Code Review Process

1. **Automated checks** must pass (style, basic tests)
2. **Maintainer review** for correctness and design
3. **Research review** for novel contributions
4. **Integration testing** to ensure compatibility

## Community Guidelines

- Be respectful and constructive in discussions
- Help others learn and contribute
- Focus on the research goals and practical applications
- Share knowledge and insights from your experiments

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and brainstorming
- **Documentation**: Check README and code comments first

## Roadmap

Current priorities for the project:

### Phase 1: Core Improvements
- [ ] Real tokenizer integration
- [ ] Comprehensive test suite
- [ ] Performance optimizations
- [ ] Better documentation

### Phase 2: Extensions
- [ ] Additional domains (SQL, math)
- [ ] Advanced evaluation benchmarks
- [ ] Transfer learning capabilities
- [ ] Visualization tools

### Phase 3: Research
- [ ] Large-scale experiments
- [ ] Comparison studies
- [ ] Novel architectural improvements
- [ ] Real-world applications

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to RI-TRM research! 🚀