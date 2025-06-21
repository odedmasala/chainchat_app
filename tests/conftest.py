import os
import tempfile
from unittest.mock import Mock, patch

import pytest

os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
os.environ["TESTING"] = "1"


@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    env_vars = {"OPENAI_API_KEY": "test-key-for-testing", "TESTING": "1"}

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Clean up (optional, as test env should be isolated)
    for key in env_vars:
        os.environ.pop(key, None)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        yield f

    # Clean up
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for testing."""
    with patch("chainchat.chat.OpenAIEmbeddings") as mock:
        mock.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
        mock.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        yield mock


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    with patch("chainchat.chat.ChatOpenAI") as mock:
        yield mock


@pytest.fixture
def mock_faiss():
    """Mock FAISS vector store for testing."""
    with patch("chainchat.chat.FAISS") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_openai_for_tests():
    """Mock OpenAI API calls and reset global chat service state for all tests."""
    from langchain.schema import AIMessage, HumanMessage

    from chainchat.chat import chat_service

    # Store original state
    original_embeddings = chat_service.embeddings
    original_llm = chat_service.llm
    original_vector_store = chat_service.vector_store
    original_documents = chat_service.documents.copy()
    original_document_sources = chat_service.document_sources.copy()
    original_sessions = chat_service.sessions.copy()

    # Create mock instances with smarter responses
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 10
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_llm = Mock()

    mock_vector_store = Mock()
    mock_vector_store.as_retriever.return_value = Mock()

    # Reset global chat service state
    chat_service.documents = []
    chat_service.document_sources = {}
    chat_service.sessions = {}
    chat_service.vector_store = None

    # Patch the global chat_service instance
    chat_service.embeddings = mock_embeddings
    chat_service.llm = mock_llm
    chat_service.embedding_type = "mock"

    # Mock the conversation chains with context-aware responses
    with (
        patch("chainchat.chat.ConversationChain") as mock_conv_chain,
        patch("chainchat.chat.ConversationalRetrievalChain") as mock_rag_chain,
        patch("chainchat.chat.ConversationBufferWindowMemory") as mock_memory_class,
        patch("chainchat.chat.FAISS") as mock_faiss,
    ):

        # Create a more realistic memory mock that tracks messages
        def create_mock_memory(*args, **kwargs):
            mock_memory = Mock()
            mock_memory.chat_memory = Mock()
            mock_memory.chat_memory.messages = []

            # Mock the memory key based on the kwargs
            if kwargs.get("memory_key") == "chat_history":
                mock_memory.memory_key = "chat_history"
                mock_memory.output_key = "answer"
            else:
                mock_memory.memory_key = "history"

            return mock_memory

        # Mock ConversationChain for direct chat
        def create_mock_conv_chain(*args, **kwargs):
            mock_conv_instance = Mock()

            def mock_predict(input):
                # Add messages to memory when predict is called
                memory = kwargs.get("memory")
                if memory and hasattr(memory, "chat_memory"):
                    memory.chat_memory.messages.append(HumanMessage(content=input))
                    memory.chat_memory.messages.append(
                        AIMessage(
                            content="Hello! I'm an AI assistant. I can help you with questions and have conversations."
                        )
                    )
                return "Hello! I'm an AI assistant. I can help you with questions and have conversations."

            mock_conv_instance.predict = mock_predict
            return mock_conv_instance

        mock_conv_chain.side_effect = create_mock_conv_chain

        # Mock ConversationalRetrievalChain for RAG - always return AI-related content
        def create_mock_rag_chain(*args, **kwargs):
            mock_rag_instance = Mock()

            def mock_invoke(input_dict):
                question = input_dict.get("question", "")

                # Add messages to memory when invoke is called
                memory = kwargs.get("memory")
                if memory and hasattr(memory, "chat_memory"):
                    memory.chat_memory.messages.append(HumanMessage(content=question))

                    # Generate different responses based on question
                    if "machine learning" in question.lower():
                        answer = "Machine Learning (ML) is a subset of AI that focuses on algorithms that improve automatically through experience. There are three main types: supervised learning, unsupervised learning, and reinforcement learning."
                    else:
                        answer = "Artificial Intelligence (AI) is a fascinating field that aims to create intelligent machines. These machines can perform tasks that typically require human intelligence, such as learning from experience, recognizing patterns, making decisions, and understanding natural language."

                    memory.chat_memory.messages.append(AIMessage(content=answer))

                    return {
                        "answer": answer,
                        "source_documents": [
                            Mock(
                                metadata={"source": "ai_document.txt", "chunk_id": 0},
                                page_content="Artificial Intelligence (AI) is a fascinating field that aims to create intelligent machines.",
                            ),
                            Mock(
                                metadata={"source": "ai_document.txt", "chunk_id": 1},
                                page_content="Machine Learning (ML) is a subset of AI that focuses on algorithms that improve automatically through experience.",
                            ),
                        ],
                    }
                else:
                    return {
                        "answer": "I don't have access to memory to store this conversation.",
                        "source_documents": [],
                    }

            mock_rag_instance.invoke = mock_invoke
            return mock_rag_instance

        mock_rag_chain.from_llm.side_effect = create_mock_rag_chain

        # Mock memory creation
        mock_memory_class.side_effect = create_mock_memory

        # Mock FAISS vector store but allow document state updates
        def mock_faiss_from_documents(documents, embeddings):
            # Update the chat service state to reflect that vector store was created
            chat_service.vector_store = mock_vector_store
            return mock_vector_store

        mock_faiss.from_documents.side_effect = mock_faiss_from_documents

        yield {
            "embeddings": mock_embeddings,
            "llm": mock_llm,
            "vector_store": mock_vector_store,
        }

    # Restore original state after test
    chat_service.embeddings = original_embeddings
    chat_service.llm = original_llm
    chat_service.vector_store = original_vector_store
    chat_service.documents = original_documents
    chat_service.document_sources = original_document_sources
    chat_service.sessions = original_sessions
