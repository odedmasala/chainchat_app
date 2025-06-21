#!/bin/bash

# ChainChat Test Runner Script
# This script runs the same tests as the CI pipeline

set -e

echo "🧪 ChainChat Test Suite Runner"
echo "==============================="
echo ""

# Check if we're in a poetry environment
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first."
    exit 1
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
poetry install --no-interaction

# Install additional test tools
poetry run pip install pytest pytest-cov pytest-html coverage

echo ""
echo "🔐 Checking API Keys..."
echo "----------------------"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Some tests may be skipped."
fi
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "⚠️  HUGGINGFACE_TOKEN not set. Some tests may be skipped."
fi
if [ -z "$PINECONE_API_KEY" ]; then
    echo "⚠️  PINECONE_API_KEY not set. Some tests may be skipped."
fi

echo ""
echo "🧪 Running Unit Tests..."
echo "------------------------"
OPENAI_API_KEY="$OPENAI_API_KEY" \
HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
PINECONE_API_KEY="$PINECONE_API_KEY" \
poetry run pytest tests/unit/ -v -m "unit" \
    --cov=chainchat \
    --cov-report=xml:coverage-unit.xml \
    --cov-report=html:htmlcov-unit/ \
    --html=unit-test-report.html \
    --self-contained-html

echo ""
echo "🔗 Running Integration Tests..."
echo "------------------------------"
OPENAI_API_KEY="$OPENAI_API_KEY" \
HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
PINECONE_API_KEY="$PINECONE_API_KEY" \
poetry run pytest tests/integration/ -v -m "integration" \
    --cov=chainchat \
    --cov-append \
    --cov-report=xml:coverage-integration.xml \
    --cov-report=html:htmlcov-integration/ \
    --html=integration-test-report.html \
    --self-contained-html

echo ""
echo "📊 Generating Combined Coverage Report..."
echo "----------------------------------------"
poetry run coverage combine
poetry run coverage xml -o coverage-combined.xml
poetry run coverage html -d htmlcov-combined/
poetry run coverage report --show-missing

echo ""
echo "✅ All tests completed successfully!"
echo ""
echo "📋 Generated Reports:"
echo "  • Unit Test Report: unit-test-report.html"
echo "  • Integration Test Report: integration-test-report.html"
echo "  • Combined Coverage: htmlcov-combined/index.html"
echo "  • Unit Coverage: htmlcov-unit/index.html"
echo "  • Integration Coverage: htmlcov-integration/index.html"
echo ""
echo "🚀 Your code is ready for CI!" 