# ChainChat AI Document Assistant

A modern AI-powered document assistant built with FastAPI, LangChain, and vector databases. ChainChat allows users to upload documents, ask questions about them, and get intelligent responses powered by Large Language Models.

[![Code Quality & Testing](https://github.com/odedmasala/chainchat_app/actions/workflows/code-quality.yml/badge.svg)](https://github.com/odedmasala/chainchat_app/actions/workflows/code-quality.yml)
[![Security Analysis](https://github.com/odedmasala/chainchat_app/actions/workflows/security-scan.yml/badge.svg)](https://github.com/odedmasala/chainchat_app/actions/workflows/security-scan.yml)
[![Vulnerability Scanning](https://github.com/odedmasala/chainchat_app/actions/workflows/vulnerability-scan.yml/badge.svg)](https://github.com/odedmasala/chainchat_app/actions/workflows/vulnerability-scan.yml)

## 🚀 Features

### Core Functionality

- **📄 Document Upload**: Support for PDF, TXT, MD, CSV, JSON, and more
- **🤖 AI Chat Interface**: Intelligent Q&A about uploaded documents
- **🌐 Multilingual Support**: Native support for English, Hebrew, and other languages
- **💾 Session Management**: Persistent chat sessions with conversation history
- **🔍 Vector Search**: Advanced semantic search using FAISS or Pinecone
- **📊 Source Citations**: Automatic source attribution for AI responses

### Technical Features

- **⚡ High Performance**: Built with FastAPI for async performance
- **🔒 Security First**: Comprehensive security scanning and vulnerability management
- **🧪 Production Ready**: Extensive testing suite with 90%+ coverage
- **📈 Scalable**: Microservices architecture with Docker support
- **🔄 CI/CD Pipeline**: Automated testing, security scanning, and deployment

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **AI/ML**: LangChain, OpenAI GPT-4, Sentence Transformers
- **Vector Database**: FAISS (local) / Pinecone (cloud)
- **Document Processing**: PyPDF, PyMuPDF, python-multipart
- **Testing**: pytest, pytest-cov, pytest-asyncio
- **Code Quality**: Black, isort, flake8, mypy
- **Security**: Bandit, Safety, Trivy, GitLeaks, TruffleHog
- **Dependency Management**: Poetry
- **Containerization**: Docker

## 📋 Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- OpenAI API key
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/odedmasala/chainchat_app.git
cd chainchat_app
```

### 2. Install Dependencies

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 3. Environment Setup

```bash
# Create environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

Required environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
HUGGINGFACE_TOKEN=your_huggingface_token_here  # Optional
PINECONE_API_KEY=your_pinecone_key_here        # Optional
```

### 4. Run the Application

```bash
# Activate virtual environment
poetry shell

# Start the development server
uvicorn chainchat.main:app --reload

# Or using Poetry directly
poetry run uvicorn chainchat.main:app --reload
```

The application will be available at `http://localhost:8000`

## 🌐 API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Chat Endpoints

```http
POST /api/chat
Content-Type: application/json

{
  "message": "What is this document about?",
  "session_id": "optional-session-id"
}
```

#### Document Upload

```http
POST /api/upload
Content-Type: multipart/form-data

file: <document-file>
```

#### Session Management

```http
GET /api/sessions/{session_id}/history
GET /api/sources
GET /api/health
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=chainchat --cov-report=html

# Run specific test categories
poetry run pytest -m unit          # Unit tests only
poetry run pytest -m integration   # Integration tests only
poetry run pytest -m "not slow"    # Skip slow tests
```

### Code Quality

```bash
# Format code
poetry run black .
poetry run isort .

# Lint code
poetry run flake8 chainchat/ tests/

# Type checking
poetry run mypy chainchat/

# Run all quality checks
./scripts/run_tests.sh
```

### Security Scanning

```bash
# Security analysis
poetry run bandit -r chainchat/

# Dependency vulnerability scan
poetry run safety check

# Check for secrets
git secrets --scan
```

## 🏗️ Project Structure

```
chainchat_app/
├── chainchat/                  # Main application package
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   ├── chat.py                # Chat service implementation
│   ├── config.py              # Configuration settings
│   ├── api/                   # API routes
│   ├── core/                  # Core functionality
│   │   ├── langchain/         # LangChain integration
│   │   ├── llm/              # LLM providers
│   │   ├── memory/           # Memory management
│   │   └── vector_store/     # Vector database
│   ├── models/               # Pydantic models
│   ├── services/             # Business logic
│   ├── static/               # Static files (HTML, CSS, JS)
│   └── utils/                # Utility functions
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── e2e/                  # End-to-end tests
│   ├── conftest.py           # Test configuration
│   └── utils/                # Test utilities
├── .github/                  # GitHub Actions workflows
│   └── workflows/            # CI/CD pipelines
├── deployment/               # Deployment configurations
│   ├── docker/               # Docker files
│   ├── kubernetes/           # K8s manifests
│   └── tilt/                 # Tilt development
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── pyproject.toml           # Poetry configuration
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## 🔒 Security Features

ChainChat implements comprehensive security measures:

### Automated Security Scanning

- **Vulnerability Scanning**: Trivy, CodeQL, OSV Scanner
- **Secret Detection**: GitLeaks, TruffleHog
- **Dependency Analysis**: Safety, Bandit
- **Code Quality**: OpenSSF Scorecard compliance

### Security Best Practices

- ✅ Input validation and sanitization
- ✅ Secure API token handling
- ✅ CORS protection
- ✅ Rate limiting (planned)
- ✅ Minimal token permissions
- ✅ Regular dependency updates

## 🚢 Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t chainchat-app .

# Run container
docker run -p 8000:8000 --env-file .env chainchat-app
```

### Production Considerations

- Set up proper environment variables
- Configure logging and monitoring
- Implement rate limiting
- Set up SSL/TLS certificates
- Configure backup for session data

## 📊 Monitoring & Observability

### Health Checks

- `GET /api/health` - Application health status
- `GET /api/sources` - Document and chunk statistics

### Metrics (Planned)

- Request/response times
- Error rates
- Document processing metrics
- AI model usage statistics

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit using conventional commits (`git commit -m 'feat: add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Convention

We use [Conventional Commits](https://conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/modifications
- `chore:` Maintenance tasks

## 📝 Testing Guide

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows

### Running Tests Locally

```bash
# Install test dependencies
poetry install --with test

# Run all tests
./scripts/run_tests.sh

# Run specific test types
poetry run pytest tests/unit/
poetry run pytest tests/integration/
```

### Test Coverage

We maintain 90%+ test coverage. View coverage reports:

```bash
poetry run pytest --cov=chainchat --cov-report=html
open htmlcov/index.html
```

## 🔧 Configuration

### Environment Variables

| Variable            | Description             | Default       | Required |
| ------------------- | ----------------------- | ------------- | -------- |
| `OPENAI_API_KEY`    | OpenAI API key          | -             | ✅       |
| `OPENAI_MODEL`      | OpenAI model to use     | `gpt-4o-mini` | ❌       |
| `MAX_TOKENS`        | Max tokens per response | `1000`        | ❌       |
| `CHUNK_SIZE`        | Document chunk size     | `1000`        | ❌       |
| `CHUNK_OVERLAP`     | Chunk overlap           | `200`         | ❌       |
| `MAX_FILE_SIZE`     | Max upload size (MB)    | `100`         | ❌       |
| `HUGGINGFACE_TOKEN` | HuggingFace API token   | -             | ❌       |
| `PINECONE_API_KEY`  | Pinecone API key        | -             | ❌       |

### Advanced Configuration

See `chainchat/config.py` for all available configuration options.

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'chainchat'**

```bash
# Ensure you're in the virtual environment
poetry shell
# Or activate manually
source .venv/bin/activate
```

**OpenAI API Errors**

- Verify your API key is correct
- Check your OpenAI account billing status
- Ensure the model is available in your region

**Document Upload Failures**

- Check file size limits (100MB default)
- Verify file format is supported
- Ensure sufficient disk space

**Memory Issues with Large Documents**

- Reduce chunk size in configuration
- Use cloud vector database (Pinecone)
- Consider document preprocessing

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the AI framework
- [FastAPI](https://github.com/tiangolo/fastapi) for the web framework
- [OpenAI](https://openai.com/) for the language models
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/odedmasala/chainchat_app/issues)
- 💬 [Discussions](https://github.com/odedmasala/chainchat_app/discussions)

---

**Built with ❤️ by [Oded Masala](https://github.com/odedmasala)**
