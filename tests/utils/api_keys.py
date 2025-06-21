"""
Utility functions for API key management in tests.
"""

import os

import pytest


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    return os.getenv("OPENAI_API_KEY", "")


def get_huggingface_token() -> str:
    """Get Hugging Face token from environment variables."""
    return os.getenv("HUGGINGFACE_TOKEN", "")


def get_pinecone_api_key() -> str:
    """Get Pinecone API key from environment variables."""
    return os.getenv("PINECONE_API_KEY", "")


def has_openai_api_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(get_openai_api_key())


def has_huggingface_token() -> bool:
    """Check if Hugging Face token is available."""
    return bool(get_huggingface_token())


def has_pinecone_api_key() -> bool:
    """Check if Pinecone API key is available."""
    return bool(get_pinecone_api_key())


def has_all_api_keys() -> bool:
    """Check if all API keys are available."""
    return has_openai_api_key() and has_huggingface_token() and has_pinecone_api_key()


# Pytest skip decorators
skip_without_openai = pytest.mark.skipif(
    not has_openai_api_key(), reason="OpenAI API key not available"
)

skip_without_huggingface = pytest.mark.skipif(
    not has_huggingface_token(), reason="Hugging Face token not available"
)

skip_without_pinecone = pytest.mark.skipif(
    not has_pinecone_api_key(), reason="Pinecone API key not available"
)

skip_without_all_apis = pytest.mark.skipif(
    not has_all_api_keys(), reason="One or more API keys not available"
)
