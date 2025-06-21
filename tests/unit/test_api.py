from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from chainchat.main import app


class TestAPIEndpoints:
    """Unit tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_chat_service(self):
        """Mock the chat service for testing."""
        with patch("chainchat.main.chat_service") as mock:
            yield mock

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_check(self, client, mock_chat_service):
        """Test the health check endpoint."""
        mock_chat_service.get_sources.return_value = {
            "total_documents": 2,
            "total_chunks": 5,
        }

        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == "ChainChat"
        assert data["documents_loaded"] == 2
        assert data["total_chunks"] == 5

    def test_get_sources(self, client, mock_chat_service):
        """Test the get sources endpoint."""
        expected_sources = {
            "documents": {"doc1": {"filename": "test.txt", "chunks": 3}},
            "total_documents": 1,
            "total_chunks": 3,
        }
        mock_chat_service.get_sources.return_value = expected_sources

        response = client.get("/api/sources")
        assert response.status_code == 200
        assert response.json() == expected_sources

    def test_chat_endpoint_success(self, client, mock_chat_service):
        """Test successful chat request."""
        mock_chat_service.ask.return_value = {
            "success": True,
            "answer": "This is a test response",
            "sources": [],
            "session_id": "test-session-id",
            "message_count": 1,
        }

        response = client.post(
            "/api/chat",
            json={"message": "Hello, how are you?", "session_id": "test-session-id"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["answer"] == "This is a test response"
        assert data["session_id"] == "test-session-id"

    def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post(
            "/api/chat", json={"message": "", "session_id": "test-session-id"}
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_chat_endpoint_whitespace_message(self, client):
        """Test chat endpoint with whitespace-only message."""
        response = client.post(
            "/api/chat", json={"message": "   ", "session_id": "test-session-id"}
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_chat_endpoint_service_error(self, client, mock_chat_service):
        """Test chat endpoint when service returns error."""
        mock_chat_service.ask.return_value = {
            "success": False,
            "message": "Service error occurred",
            "answer": "I encountered an error",
        }

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["message"] == "Service error occurred"
        assert data["answer"] == "I encountered an error"

    def test_upload_endpoint_success(self, client, mock_chat_service):
        """Test successful file upload."""
        mock_chat_service.add_document.return_value = {
            "success": True,
            "message": "Document processed into 3 chunks",
            "document_id": "test-doc-id",
            "chunks": 3,
        }

        test_content = b"This is a test document content."

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", BytesIO(test_content), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks"] == 3
        assert data["document_id"] == "test-doc-id"

    def test_upload_endpoint_no_file(self, client):
        """Test upload endpoint with no file provided."""
        response = client.post("/api/upload", files={})
        assert response.status_code == 422  # Validation error

    def test_upload_endpoint_unsupported_file_type(self, client):
        """Test upload endpoint with unsupported file type."""
        test_content = b"This is a test file."

        response = client.post(
            "/api/upload",
            files={
                "file": ("test.exe", BytesIO(test_content), "application/octet-stream")
            },
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_endpoint_large_file(self, client):
        """Test upload endpoint with file that's too large."""
        # Create a large file content (larger than max_file_size)
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB

        response = client.post(
            "/api/upload",
            files={"file": ("large_test.txt", BytesIO(large_content), "text/plain")},
        )

        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_upload_endpoint_pdf_success(self, client, mock_chat_service):
        """Test successful PDF upload."""
        mock_chat_service.add_document.return_value = {
            "success": True,
            "message": "Document processed into 5 chunks",
            "document_id": "pdf-doc-id",
            "chunks": 5,
        }

        # Mock PDF content
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

        with patch("chainchat.main.extract_pdf_text") as mock_extract:
            mock_extract.return_value = "Extracted PDF text content"

            response = client.post(
                "/api/upload",
                files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["chunks"] == 5

    def test_upload_endpoint_pdf_extraction_error(self, client):
        """Test PDF upload with extraction error."""
        pdf_content = b"invalid pdf content"

        with patch("chainchat.main.extract_pdf_text") as mock_extract:
            mock_extract.side_effect = ValueError("Failed to extract text")

            response = client.post(
                "/api/upload",
                files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
            )

            assert response.status_code == 400
            assert "Error processing file" in response.json()["detail"]

    def test_upload_endpoint_unicode_error(self, client):
        """Test upload endpoint with invalid UTF-8 content."""
        # Invalid UTF-8 bytes
        invalid_content = b"\xff\xfe\x00\x00"

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", BytesIO(invalid_content), "text/plain")},
        )

        assert response.status_code == 400
        assert "UTF-8" in response.json()["detail"]

    def test_upload_endpoint_service_error(self, client, mock_chat_service):
        """Test upload endpoint when service returns error."""
        mock_chat_service.add_document.return_value = {
            "success": False,
            "message": "Document processing failed",
        }

        test_content = b"This is a test document."

        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", BytesIO(test_content), "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Document processing failed"

    def test_session_history_success(self, client, mock_chat_service):
        """Test successful session history retrieval."""
        mock_chat_service.get_session_history.return_value = {
            "success": True,
            "session_id": "test-session",
            "created_at": "2024-01-01T00:00:00",
            "message_count": 2,
            "messages": [
                {"type": "humanmessage", "content": "Hello"},
                {"type": "aimessage", "content": "Hi there!"},
            ],
        }

        response = client.get("/api/sessions/test-session/history")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["session_id"] == "test-session"
        assert len(data["messages"]) == 2

    def test_session_history_not_found(self, client, mock_chat_service):
        """Test session history for non-existent session."""
        mock_chat_service.get_session_history.return_value = {
            "success": False,
            "message": "Session not found",
        }

        response = client.get("/api/sessions/non-existent/history")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_chat_endpoint_without_session_id(self, client, mock_chat_service):
        """Test chat endpoint without providing session ID."""
        mock_chat_service.ask.return_value = {
            "success": True,
            "answer": "Hello there!",
            "sources": [],
            "session_id": "new-session-id",
            "message_count": 1,
        }

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data
        mock_chat_service.ask.assert_called_once_with("Hello", None)

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/api/chat")
        # Note: TestClient doesn't fully simulate CORS, but we can check basic setup
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented

    def test_multiple_file_types(self, client, mock_chat_service):
        """Test upload with different supported file types."""
        mock_chat_service.add_document.return_value = {
            "success": True,
            "message": "Document processed",
            "document_id": "test-id",
            "chunks": 1,
        }

        file_types = [
            ("test.txt", "text/plain"),
            ("test.md", "text/markdown"),
            ("test.csv", "text/csv"),
            ("test.json", "application/json"),
            ("test.py", "text/x-python"),
            ("test.js", "application/javascript"),
            ("test.html", "text/html"),
            ("test.css", "text/css"),
        ]

        for filename, content_type in file_types:
            test_content = b"Test content"
            response = client.post(
                "/api/upload",
                files={"file": (filename, BytesIO(test_content), content_type)},
            )
            assert response.status_code == 200, f"Failed for {filename}"
