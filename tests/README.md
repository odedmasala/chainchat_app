# Testing Guide

This directory contains the test suite for the ChainChat application.

## Test Structure

```
tests/
├── unit/           # Unit tests (isolated component testing)
├── integration/    # Integration tests (multi-component testing)
├── e2e/           # End-to-end tests (full application testing)
├── utils/         # Test utilities and helpers
└── conftest.py    # Pytest configuration and fixtures
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests that test individual components in isolation
- `@pytest.mark.integration` - Integration tests that test multiple components together
- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.requires_openai` - Tests that require OpenAI API key
- `@pytest.mark.requires_huggingface` - Tests that require Hugging Face token
- `@pytest.mark.requires_pinecone` - Tests that require Pinecone API key
- `@pytest.mark.requires_all_apis` - Tests that require all external API keys

## API Keys and Environment Variables

The test suite requires several API keys for external services:

- `OPENAI_API_KEY` - OpenAI API key for LLM interactions
- `HUGGINGFACE_TOKEN` - Hugging Face token for model access
- `PINECONE_API_KEY` - Pinecone API key for vector database

### Using API Key Utilities

Use the utilities from `tests.utils` to handle API keys gracefully:

```python
from tests.utils import skip_without_openai, skip_without_all_apis

@skip_without_openai
def test_openai_integration():
    """Test that requires OpenAI API key."""
    # This test will be skipped if OPENAI_API_KEY is not set
    pass

@skip_without_all_apis
def test_full_integration():
    """Test that requires all API keys."""
    # This test will be skipped if any API key is missing
    pass
```

## Running Tests

### Local Development

Use the provided script to run tests with the same configuration as CI:

```bash
# Run all tests
./scripts/run_tests.sh

# Or run specific test categories
poetry run pytest tests/unit/ -m "unit"
poetry run pytest tests/integration/ -m "integration"
```

### Environment Setup

1. Set your API keys in your environment:

```bash
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACE_TOKEN="your-huggingface-token"
export PINECONE_API_KEY="your-pinecone-key"
```

2. Or create a `.env` file (not tracked in git):

```env
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-huggingface-token
PINECONE_API_KEY=your-pinecone-key
```

### CI/CD Pipeline

The GitHub Actions pipeline automatically runs:

1. **Unit Tests** - Fast, isolated component tests
2. **Integration Tests** - Multi-component interaction tests
3. **Coverage Analysis** - Code coverage reporting
4. **Test Reports** - HTML reports available as artifacts

## Test Coverage

The test suite generates multiple coverage reports:

- `coverage-unit.xml` - Unit test coverage
- `coverage-integration.xml` - Integration test coverage
- `coverage-combined.xml` - Combined coverage report
- `htmlcov-*/` - HTML coverage reports for browsing

## Best Practices

1. **Unit Tests**:

   - Mock external dependencies
   - Test single components in isolation
   - Fast execution (< 1 second per test)

2. **Integration Tests**:

   - Use real external services when possible
   - Test component interactions
   - May require API keys

3. **API Key Handling**:

   - Always use the skip decorators for external service tests
   - Never hardcode API keys in test files
   - Use the utilities from `tests.utils` for consistent behavior

4. **Test Markers**:
   - Always mark tests with appropriate categories
   - Use `@pytest.mark.slow` for tests > 5 seconds
   - Mark external service dependencies clearly
