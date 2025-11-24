# Contributing to HAMHA + LMA

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/hamha-lma.git
   cd hamha-lma
   ```

3. Install development dependencies:
   ```bash
   make install-dev
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- We use **Black** for Python code formatting
- We use **Flake8** for linting
- We use **MyPy** for type checking

Run all checks:
```bash
make format  # Format code
make lint    # Check code quality
```

## Testing

All contributions must include tests:

```bash
make test         # Run all tests with coverage
make test-fast    # Run tests without coverage (faster)
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names: `test_<functionality>_<condition>_<expected_result>`
- Aim for >80% code coverage
- Test both success and failure cases

Example:
```python
def test_hamha_forward_with_valid_input_returns_correct_shape():
    model = HexagonalMultiHeadAttention(128, 2)
    x = torch.randn(2, 32, 128)
    output = model(x)
    assert output.shape == (2, 32, 128)
```

## Documentation

- Use docstrings for all public classes and functions
- Follow Google-style docstring format:
  ```python
  def function(arg1: int, arg2: str) -> bool:
      \"\"\"Brief description.

      Longer description if needed.

      Args:
          arg1: Description of arg1
          arg2: Description of arg2

      Returns:
          Description of return value

      Raises:
          ValueError: When to raise this error
      \"\"\"
  ```

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit pull request with clear description

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

## Commit Messages

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(lma): Add predictive entropy monitoring

Implement time-series forecasting for attention entropy
values to predict potential fixation events before they occur.

Closes #42
```

## Issue Reporting

When reporting issues, include:
- Python version
- PyTorch version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/stack traces

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Clearly describe the use case
- Provide examples if possible
- Explain why it benefits the project

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers learn

## Questions?

- Open a discussion on GitHub Discussions
- Check existing documentation
- Review closed issues for similar questions

Thank you for contributing to HAMHA + LMA! 🔷
