# Contributing to UIDAI Risk Intelligence System

Thank you for your interest in contributing to this project!

## Development Guidelines

1. **Coding Style**: Follow PEP 8 guidelines. Add docstrings, inline comments, and type annotations for all new modules.
2. **Branching Model**:
   - Create feature branches from `main` (e.g. `feature/new-anomaly-score`).
   - Open a PR targeting `main`.
3. **Tests**:
   - Ensure all tests pass.
   - Run tests locally using `pytest tests/`.
   - Write unit tests for new features.
4. **Logs**:
   - Write logs using the system logging module `src.utils.logger`.
   - Avoid printing to standard out for pipeline runs.
5. **Configuration**:
   - Never hardcode file paths or hyperparameters.
   - Store settings inside `config/config.yaml`.
