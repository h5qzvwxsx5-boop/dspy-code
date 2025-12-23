# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**DSPy Code** is an AI-powered interactive development environment (IDE) for building and optimizing DSPy applications. It provides a specialized CLI that helps developers:

- Develop DSPy applications using natural language
- Optimize with GEPA (Genetic Pareto) workflows
- Validate code against DSPy best practices
- Deploy production-ready applications using MCP integration

**Status:** Beta (v0.1.5)
**Python:** 3.10, 3.11, 3.12, 3.13

## Project Structure

```
dspy-code/
├── dspy_code/              # Main package
│   ├── commands/           # Interactive slash commands
│   ├── core/               # Core utilities and exceptions
│   ├── execution/          # Code execution engine
│   ├── export/             # Export/import workflows
│   ├── generators/         # Code generators
│   ├── mcp/                # Model Context Protocol client
│   ├── models/             # LLM connection handling
│   ├── optimization/       # GEPA optimization workflows
│   ├── project/            # Project management
│   ├── rag/                # Retrieval-Augmented Generation indexing
│   ├── session/            # Session state management
│   ├── templates/          # Pre-built DSPy patterns (20+)
│   ├── ui/                 # Terminal UI components
│   ├── validation/         # DSPy code validation
│   └── main.py             # CLI entry point
├── tests/                  # Test suite
├── docs/                   # MkDocs documentation
└── examples/               # Example scripts
```

## Development Commands

### Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dspy_code --cov-report=html

# Run specific test file
pytest tests/test_specific_file.py

# Run in parallel
pytest -n auto

# Run by marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
```

### Linting and Formatting

```bash
# Run linter
ruff check .

# Fix linting issues
ruff check --fix .

# Run formatter
ruff format .

# Type checking
mypy dspy_code
```

### Building

```bash
# Build package
python -m build

# Check package
twine check dist/*
```

## Code Style

- **Linter/Formatter:** Ruff
- **Line length:** 100 characters
- **Type hints:** Required for function signatures
- **Docstrings:** Google-style format
- **Imports:** Organized by Ruff (stdlib, third-party, local)

## Commit Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Formatting changes
- `refactor:` - Code restructuring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## Key Dependencies

- **dspy>=3.0.4** - Core DSPy framework
- **click>=8.0.0** - CLI framework
- **rich>=13.7.0** - Terminal UI
- **mcp>=1.2.1** - Model Context Protocol
- **pydantic>=2.11.0** - Data validation

## Important Notes

- This project uses `hatchling` as the build backend (no setup.py)
- Entry point: `dspy-code` command maps to `dspy_code.main:main`
- Tests use pytest with markers: `unit`, `integration`, `slow`
- Pre-commit hooks enforce quality standards
- CI requires passing all Python versions (3.10-3.13)

## Documentation

- **Main docs:** `docs/` directory (MkDocs)
- **API reference:** Docstrings in source code
- **Changelog:** `CHANGELOG.md` (Keep a Changelog format)
- **Contributing:** `CONTRIBUTING.md`

## LLM Providers

Optional dependencies for different providers:
- `dspy-code[openai]` - OpenAI support
- `dspy-code[anthropic]` - Anthropic support
- `dspy-code[gemini]` - Google Gemini support
- `dspy-code[llm-all]` - All providers
