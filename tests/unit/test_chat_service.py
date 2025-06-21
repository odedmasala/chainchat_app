import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from chainchat.chat import ChatService
from chainchat.config import Settings


class TestChatService:
    """Unit tests for ChatService class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.openai_api_key = "test-key"
        settings.openai_model = "gpt-4o-mini"
        settings.chunk_size = 1000
        settings.chunk_overlap = 200
        return settings

    @pytest.fixture
    def chat_service(self, mock_settings):
        """Create a ChatService instance with mocked dependencies."""
        with patch('chainchat.chat.settings', mock_settings), \
             patch('chainchat.chat.OpenAIEmbeddings') as mock_embeddings, \
             patch('chainchat.chat.ChatOpenAI') as mock_llm, \
             patch('chainchat.chat.RecursiveCharacterTextSplitter') as mock_splitter:
            
            # Mock the embeddings
            mock_embeddings.return_value = Mock()
            
            # Mock the LLM
            mock_llm.return_value = Mock()
            
            # Mock the text splitter
            mock_splitter.return_value = Mock()
            
            service = ChatService()
            service.embeddings = Mock()
            service.llm = Mock()
            service.text_splitter = Mock()
            
            return service

    def test_init_creates_proper_instances(self, mock_settings):
        """Test that ChatService initializes with proper instances."""
        with patch('chainchat.chat.settings', mock_settings), \
             patch('chainchat.chat.OpenAIEmbeddings') as mock_embeddings, \
             patch('chainchat.chat.ChatOpenAI') as mock_llm, \
             patch('chainchat.chat.RecursiveCharacterTextSplitter') as mock_splitter:
            
            service = ChatService()
            
            assert service.documents == []
            assert service.vector_store is None
            assert service.document_sources == {}
            assert service.sessions == {}

    def test_add_document_success(self, chat_service):
        """Test successful document addition."""
        test_text = "This is a test document with some content."
        test_filename = "test.txt"
        
        # Mock text splitter
        chat_service.text_splitter.split_text.return_value = [
            "This is a test document",
            "with some content."
        ]
        
        # Mock vector store creation
        with patch('chainchat.chat.FAISS') as mock_faiss:
            mock_faiss.from_documents.return_value = Mock()
            
            result = chat_service.add_document(test_text, test_filename)
            
            assert result["success"] is True
            assert result["chunks"] == 2
            assert "document_id" in result
            assert len(chat_service.documents) == 2

    def test_add_document_duplicate(self, chat_service):
        """Test adding duplicate document."""
        test_text = "This is a test document."
        test_filename = "test.txt"
        
        # Add document first time
        chat_service.text_splitter.split_text.return_value = ["This is a test document."]
        
        with patch('chainchat.chat.FAISS') as mock_faiss:
            mock_faiss.from_documents.return_value = Mock()
            
            # First addition
            result1 = chat_service.add_document(test_text, test_filename)
            assert result1["success"] is True
            
            # Second addition (duplicate)
            result2 = chat_service.add_document(test_text, test_filename)
            assert result2["success"] is False
            assert "already exists" in result2["message"]

    def test_add_document_error_handling(self, chat_service):
        """Test error handling in document addition."""
        test_text = "This is a test document."
        test_filename = "test.txt"
        
        # Mock text splitter to raise exception
        chat_service.text_splitter.split_text.side_effect = Exception("Test error")
        
        result = chat_service.add_document(test_text, test_filename)
        
        assert result["success"] is False
        assert "Error processing document" in result["message"]

    def test_ask_direct_chat_mode(self, chat_service):
        """Test asking questions in direct chat mode (no documents)."""
        question = "What is artificial intelligence?"
        
        # Mock conversation chain
        with patch('chainchat.chat.ConversationChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.predict.return_value = "AI is a field of computer science."
            mock_chain_class.return_value = mock_chain
            
            # Mock memory
            with patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
                mock_memory = Mock()
                mock_memory_class.return_value = mock_memory
                
                result = chat_service.ask(question)
                
                assert result["success"] is True
                assert result["answer"] == "AI is a field of computer science."
                assert result["sources"] == []
                assert result["mode"] == "direct_chat"
                assert "session_id" in result

    def test_ask_rag_mode(self, chat_service):
        """Test asking questions in RAG mode (with documents)."""
        question = "What is in the document?"
        
        # Set up vector store
        chat_service.vector_store = Mock()
        
        # Mock ConversationalRetrievalChain
        with patch('chainchat.chat.ConversationalRetrievalChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "answer": "The document contains information about AI.",
                "source_documents": [
                    Mock(
                        metadata={"source": "test.pdf", "chunk_id": 0},
                        page_content="This is test content from the document."
                    )
                ]
            }
            mock_chain_class.from_llm.return_value = mock_chain
            
            # Mock memory
            with patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
                mock_memory = Mock()
                mock_memory.chat_memory = Mock()
                mock_memory.chat_memory.messages = []
                mock_memory_class.return_value = mock_memory
                
                result = chat_service.ask(question)
                
                assert result["success"] is True
                assert result["answer"] == "The document contains information about AI."
                assert len(result["sources"]) > 0
                assert result["mode"] == "rag_chat"

    def test_ask_with_session_id(self, chat_service):
        """Test asking questions with specific session ID."""
        question = "Hello"
        session_id = str(uuid.uuid4())
        
        with patch('chainchat.chat.ConversationChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.predict.return_value = "Hello! How can I help you?"
            mock_chain_class.return_value = mock_chain
            
            with patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
                mock_memory = Mock()
                mock_memory_class.return_value = mock_memory
                
                result = chat_service.ask(question, session_id)
                
                assert result["success"] is True
                assert result["session_id"] == session_id
                assert session_id in chat_service.sessions

    def test_ask_openai_quota_exceeded(self, chat_service):
        """Test handling OpenAI quota exceeded error."""
        question = "What is AI?"
        
        with patch('chainchat.chat.ConversationChain') as mock_chain_class:
            mock_chain = Mock()
            mock_chain.predict.side_effect = Exception("quota exceeded")
            mock_chain_class.return_value = mock_chain
            
            with patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
                mock_memory = Mock()
                mock_memory_class.return_value = mock_memory
                
                result = chat_service.ask(question)
                
                assert result["success"] is False
                assert "quota" in result["message"].lower()
                assert "OpenAI API Quota Exceeded" in result["answer"]

    def test_get_sources_empty(self, chat_service):
        """Test getting sources when no documents are loaded."""
        result = chat_service.get_sources()
        
        assert result["documents"] == {}
        assert result["total_documents"] == 0
        assert result["total_chunks"] == 0

    def test_get_sources_with_documents(self, chat_service):
        """Test getting sources with loaded documents."""
        # Add some mock document sources
        chat_service.document_sources = {
            "doc1": {"filename": "test1.txt", "chunks": 2},
            "doc2": {"filename": "test2.txt", "chunks": 3}
        }
        chat_service.documents = [Mock()] * 5  # 5 total chunks
        
        result = chat_service.get_sources()
        
        assert result["total_documents"] == 2
        assert result["total_chunks"] == 5
        assert "doc1" in result["documents"]
        assert "doc2" in result["documents"]

    def test_get_session_history_not_found(self, chat_service):
        """Test getting session history for non-existent session."""
        result = chat_service.get_session_history("non-existent-session")
        
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_get_session_history_success(self, chat_service):
        """Test getting session history successfully."""
        session_id = str(uuid.uuid4())
        
        # Create mock session
        mock_memory = Mock()
        mock_memory.chat_memory = Mock()
        mock_memory.chat_memory.messages = [
            Mock(__class__=Mock(__name__="HumanMessage"), content="Hello"),
            Mock(__class__=Mock(__name__="AIMessage"), content="Hi there!")
        ]
        
        chat_service.sessions[session_id] = {
            "memory": mock_memory,
            "created_at": datetime.now().isoformat(),
            "message_count": 2
        }
        
        result = chat_service.get_session_history(session_id)
        
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert len(result["messages"]) == 2
        assert result["message_count"] == 2

    def test_rebuild_vector_store_success(self, chat_service):
        """Test successful vector store rebuild."""
        # Add some documents
        chat_service.documents = [Mock(), Mock()]
        
        with patch('chainchat.chat.FAISS') as mock_faiss:
            mock_vector_store = Mock()
            mock_faiss.from_documents.return_value = mock_vector_store
            
            chat_service._rebuild_vector_store()
            
            assert chat_service.vector_store == mock_vector_store
            mock_faiss.from_documents.assert_called_once_with(
                chat_service.documents,
                chat_service.embeddings
            )

    def test_rebuild_vector_store_openai_quota_fallback(self, chat_service):
        """Test vector store rebuild with OpenAI quota exceeded fallback."""
        chat_service.documents = [Mock(), Mock()]
        
        with patch('chainchat.chat.FAISS') as mock_faiss, \
             patch('chainchat.chat.SentenceTransformerEmbeddings') as mock_local_embeddings:
            
            # First call fails with quota error
            mock_faiss.from_documents.side_effect = [
                Exception("quota exceeded"),
                Mock()  # Second call succeeds
            ]
            
            mock_local_embeddings.return_value = Mock()
            
            chat_service._rebuild_vector_store()
            
            # Should have switched to local embeddings
            assert chat_service.embedding_type == "local"
            assert mock_faiss.from_documents.call_count == 2

    def test_memory_creation_direct_chat(self, chat_service):
        """Test memory creation for direct chat mode."""
        question = "Hello"
        
        with patch('chainchat.chat.ConversationChain') as mock_chain_class, \
             patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
            
            mock_chain = Mock()
            mock_chain.predict.return_value = "Hi!"
            mock_chain_class.return_value = mock_chain
            
            mock_memory = Mock()
            mock_memory_class.return_value = mock_memory
            
            result = chat_service.ask(question)
            
            # Check that memory was created with correct parameters for direct chat
            mock_memory_class.assert_called_with(
                memory_key="history",
                return_messages=True,
                k=5
            )

    def test_memory_creation_rag_mode(self, chat_service):
        """Test memory creation for RAG mode."""
        question = "What's in the document?"
        chat_service.vector_store = Mock()
        
        with patch('chainchat.chat.ConversationalRetrievalChain') as mock_chain_class, \
             patch('chainchat.chat.ConversationBufferWindowMemory') as mock_memory_class:
            
            mock_chain = Mock()
            mock_chain.invoke.return_value = {"answer": "Test answer", "source_documents": []}
            mock_chain_class.from_llm.return_value = mock_chain
            
            mock_memory = Mock()
            mock_memory.chat_memory = Mock()
            mock_memory.chat_memory.messages = []
            mock_memory_class.return_value = mock_memory
            
            result = chat_service.ask(question)
            
            # Check that memory was created with correct parameters for RAG
            mock_memory_class.assert_called_with(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=5
            ) 