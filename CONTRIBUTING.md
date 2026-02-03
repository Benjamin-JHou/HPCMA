# Contributing to MMRP Clinical AI

First off, thank you for considering contributing to the Hypertension Pan-Comorbidity Multi-Modal Risk Prediction project! It's people like you that make this research possible.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:
- **Scientific rigor**: All contributions must maintain high standards of scientific validity
- **Patient safety**: No code that could harm patients will be accepted
- **Transparency**: All methods and limitations must be clearly documented
- **Inclusivity**: We welcome contributors from diverse backgrounds and institutions

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, please include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (e.g., sample data that triggers the issue)
- **Describe the behavior you observed** and what behavior you expected
- **Include system information** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples** to demonstrate the enhancement
- **Explain why this enhancement would be useful**

### Contributing Code

#### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/Benjamin-JHou/MMRP-Clinical-AI.git
   cd MMRP-Clinical-AI
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

#### Coding Standards

- **Python 3.9+** is required
- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Write **docstrings** for all public functions and classes
- Maintain **test coverage** > 80%

#### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run tests
pytest tests/ --cov=src --cov-report=html
```

#### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pre-commit install
```

These hooks will automatically:
- Format code with Black
- Sort imports with isort
- Run basic linting checks

#### Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Update documentation** (README, docstrings, etc.)

5. **Ensure all checks pass**:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   pytest tests/
   ```

6. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "feat: add support for new disease risk prediction"
   ```

   We follow conventional commits:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding or updating tests
   - `refactor:` Code refactoring
   - `style:` Code style changes (formatting)
   - `chore:` Maintenance tasks

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist of completed tasks

#### PR Review Criteria

Your PR will be reviewed for:
- **Code quality** (readability, maintainability)
- **Test coverage** (new code must have tests)
- **Documentation** (docstrings, README updates)
- **Scientific validity** (methods are sound and validated)
- **No breaking changes** (unless discussed and approved)

## Research Contributions

For research contributions (new methods, disease models, etc.):

1. **Include validation results** showing model performance
2. **Provide documentation** on methodology and assumptions
3. **Add to the atlas** by updating relevant data files
4. **Cite relevant literature** supporting your approach
5. **Consider clinical implications** of your contributions

## Specific Contribution Areas

### High Priority

- External validation in diverse populations
- Additional disease models
- Enhanced fairness/bias mitigation
- Performance optimization
- Documentation improvements

### Medium Priority

- Additional visualization tools
- Web interface improvements
- Additional export formats
- CI/CD enhancements

### Documentation

- Tutorials and examples
- API documentation
- Clinical integration guides
- Methodology documentation

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Contact the maintainers

## Recognition

Contributors will be:
- Listed in the README contributors section
- Acknowledged in release notes
- Co-authors on relevant publications (if significant contribution)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to precision medicine research!**
