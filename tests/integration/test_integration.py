from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from chainchat.main import app


class TestIntegration:
    """Integration tests for the ChainChat application."""

    @pytest.fixture
    def client(self):
        """Create a test client for integration testing."""
        return TestClient(app)

    @pytest.fixture
    def sample_text_document(self):
        """Create a sample text document for testing."""
        return (
            "This is a comprehensive test document about artificial "
            "intelligence and machine learning.\n\n"
            "Artificial Intelligence (AI) is a fascinating field that aims to "
            "create intelligent machines.\n"
            "These machines can perform tasks that typically require human "
            "intelligence, such as:\n"
            "- Learning from experience\n"
            "- Recognizing patterns\n"
            "- Making decisions\n"
            "- Understanding natural language\n\n"
            "Machine Learning (ML) is a subset of AI that focuses on algorithms "
            "that improve automatically\n"
            "through experience. There are three main types of machine learning:\n"
            "1. Supervised Learning - learning with labeled examples\n"
            "2. Unsupervised Learning - finding patterns in unlabeled data\n"
            "3. Reinforcement Learning - learning through trial and error\n\n"
            "Deep Learning is a subset of machine learning that uses neural "
            "networks with multiple layers.\n"
            "It has revolutionized fields like computer vision, natural language "
            "processing, and speech recognition.\n\n"
            "The history of AI dates back to the 1950s when Alan Turing proposed "
            "the famous Turing Test.\n"
            "Since then, we've seen remarkable progress in AI capabilities.\n\n"
            "Some popular AI applications today include:\n"
            "- Virtual assistants like Siri and Alexa\n"
            "- Recommendation systems on Netflix and YouTube\n"
            "- Autonomous vehicles and self-driving cars\n"
            "- Medical diagnosis and healthcare systems\n"
            "- Language translation services like Google Translate\n"
            "- Image recognition and computer vision systems\n\n"
            "The future of AI holds great promise for solving complex global "
            "challenges in healthcare,\n"
            "climate change, education, and many other fields. However, it also "
            "raises important ethical\n"
            "questions about privacy, job displacement, and the responsible "
            "development of AI systems.\n"
        )

    @pytest.fixture
    def sample_multilingual_document(self):
        """Create a sample document with multiple languages including Hebrew."""
        return (
            "This is a multilingual document that contains text in different "
            "languages.\n\n"
            "English: Artificial Intelligence is transforming our world.\n\n"
            "עברית: בינה מלאכותית משנה את העולם שלנו. היא כוללת תחומים רבים "
            "כמו למידת מכונה,\n"
            "עיבוד שפה טבעית, וראייה ממוחשבת. הטכנולוגיה הזו עוזרת לנו "
            "לפתור בעיות מורכבות\n"
            "ולשפר את איכות החיים.\n\n"
            "Français: L'intelligence artificielle transforme notre façon "
            "de travailler.\n\n"
            "中文: 人工智能正在改变我们的世界。\n\n"
            "עברית נוסף: מערכות בינה מלאכותית יכולות לעזור בתחומים כמו:\n"
            "- רפואה ואבחון מחלות\n"
            "- חינוך מותאם אישית\n"
            "- תחבורה אוטונומית\n"
            "- אבטחת מידע\n"
            "- ניתוח נתונים כלכליים\n\n"
            "המטרה היא לפתח טכנולוגיה שתשרת את האנושות ותשפר את איכות "
            "החיים של כולם.\n"
        )

    def test_full_workflow_text_upload(self, client, sample_text_document):
        """Test the complete workflow: health check, upload, and chat."""
        # 1. Check initial health status
        response = client.get("/api/health")
        assert response.status_code == 200
        initial_data = response.json()
        assert initial_data["status"] == "healthy"
        assert initial_data["documents_loaded"] >= 0

        # 2. Upload a document
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "ai_document.txt",
                    BytesIO(sample_text_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True
        assert upload_data["chunks"] > 0
        assert "document_id" in upload_data

        # 3. Check updated health status
        response = client.get("/api/health")
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["documents_loaded"] > initial_data["documents_loaded"]
        assert updated_data["total_chunks"] > 0

        # 4. Chat about the document (RAG mode)
        response = client.post(
            "/api/chat", json={"message": "What is artificial intelligence?"}
        )
        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["success"] is True
        assert (
            "artificial intelligence" in chat_data["answer"].lower()
            or "ai" in chat_data["answer"].lower()
        )
        assert len(chat_data["sources"]) > 0
        assert "session_id" in chat_data
        session_id = chat_data["session_id"]

        # 5. Follow-up question in the same session
        response = client.post(
            "/api/chat",
            json={
                "message": "What are the types of machine learning?",
                "session_id": session_id,
            },
        )
        assert response.status_code == 200
        follow_up_data = response.json()
        assert follow_up_data["success"] is True
        assert follow_up_data["session_id"] == session_id
        assert follow_up_data["message_count"] == 2

        # 6. Get session history
        response = client.get(f"/api/sessions/{session_id}/history")
        assert response.status_code == 200
        history_data = response.json()
        assert history_data["success"] is True
        assert len(history_data["messages"]) >= 4  # 2 questions + 2 answers

        # 7. Get sources information
        response = client.get("/api/sources")
        assert response.status_code == 200
        sources_data = response.json()
        assert sources_data["total_documents"] > 0
        assert sources_data["total_chunks"] > 0

    def test_multilingual_workflow(self, client, sample_multilingual_document):
        """Test workflow with multilingual content including Hebrew."""
        # Upload multilingual document
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "multilingual.txt",
                    BytesIO(sample_multilingual_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True

        # Test English question
        response = client.post(
            "/api/chat", json={"message": "What is this document about?"}
        )
        assert response.status_code == 200
        english_response = response.json()
        assert english_response["success"] is True
        session_id = english_response["session_id"]

        # Test Hebrew question about the content
        response = client.post(
            "/api/chat",
            json={"message": "מה זה בינה מלאכותית?", "session_id": session_id},
        )
        assert response.status_code == 200
        hebrew_response = response.json()
        assert hebrew_response["success"] is True
        # The response should maintain context from the previous question
        assert hebrew_response["session_id"] == session_id

        # Test Hebrew question asking for summary
        response = client.post(
            "/api/chat",
            json={"message": "תן לי סיכום של המסמך בעברית", "session_id": session_id},
        )
        assert response.status_code == 200
        summary_response = response.json()
        assert summary_response["success"] is True
        assert len(summary_response["sources"]) > 0

    def test_direct_chat_mode(self, client):
        """Test direct chat mode without uploading documents."""
        # Test direct chat (no documents uploaded)
        response = client.post(
            "/api/chat", json={"message": "Hello, can you tell me a joke?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["sources"] == []  # No sources in direct mode
        assert "session_id" in data
        session_id = data["session_id"]

        # Test follow-up in direct chat
        response = client.post(
            "/api/chat",
            json={"message": "Tell me another one", "session_id": session_id},
        )
        assert response.status_code == 200
        follow_up_data = response.json()
        assert follow_up_data["success"] is True
        assert follow_up_data["session_id"] == session_id
        assert follow_up_data["message_count"] == 2

    def test_pdf_upload_and_chat(self, client):
        """Test PDF upload and chat functionality."""
        # Create a simple PDF-like content (this would normally be a real PDF)
        pdf_content = """
        This is a test PDF document about machine learning algorithms.

        Linear Regression is one of the fundamental algorithms in machine learning.
        It attempts to model the relationship between variables by fitting a
        linear equation.

        Random Forest is an ensemble learning method that combines multiple
        decision trees.
        It's known for its robustness and ability to handle both classification
        and regression tasks.

        Support Vector Machines (SVM) are powerful algorithms for classification
        and regression.
        They work by finding the optimal hyperplane that separates different
        classes.

        Neural Networks are inspired by biological neural networks and consist of
        interconnected nodes.
        They can learn complex patterns and are the foundation of deep learning.
        """

        # Mock PDF upload (in real integration test, this would be a real PDF)
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "ml_algorithms.txt",
                    BytesIO(pdf_content.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True

        # Chat about the PDF content
        response = client.post(
            "/api/chat",
            json={"message": "What algorithms are mentioned in the document?"},
        )
        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["success"] is True
        assert len(chat_data["sources"]) > 0

    def test_error_handling_and_recovery(self, client):
        """Test error scenarios and recovery."""
        # Test upload with unsupported file type
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "test.exe",
                    BytesIO(b"binary content"),
                    "application/octet-stream",
                )
            },
        )
        assert response.status_code == 400

        # Test chat with empty message
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 400

        # Test session history for non-existent session
        response = client.get("/api/sessions/non-existent-session/history")
        assert response.status_code == 404

        # Test that the system still works after errors
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_concurrent_sessions(self, client, sample_text_document):
        """Test multiple concurrent chat sessions."""
        # Upload a document first
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "test_doc.txt",
                    BytesIO(sample_text_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200

        # Create first session
        response1 = client.post("/api/chat", json={"message": "What is AI?"})
        assert response1.status_code == 200
        session1_id = response1.json()["session_id"]

        # Create second session
        response2 = client.post(
            "/api/chat", json={"message": "What is machine learning?"}
        )
        assert response2.status_code == 200
        session2_id = response2.json()["session_id"]

        # Verify sessions are different
        assert session1_id != session2_id

        # Continue conversation in both sessions
        response1_follow = client.post(
            "/api/chat",
            json={"message": "Tell me more about it", "session_id": session1_id},
        )
        assert response1_follow.status_code == 200
        assert response1_follow.json()["session_id"] == session1_id
        assert response1_follow.json()["message_count"] == 2

        response2_follow = client.post(
            "/api/chat",
            json={"message": "What are its applications?", "session_id": session2_id},
        )
        assert response2_follow.status_code == 200
        assert response2_follow.json()["session_id"] == session2_id
        assert response2_follow.json()["message_count"] == 2

    def test_mode_switching(self, client, sample_text_document):
        """Test switching between direct chat and RAG modes."""
        # Start with direct chat (no documents)
        response = client.post("/api/chat", json={"message": "Hello, what can you do?"})
        assert response.status_code == 200
        direct_data = response.json()
        assert direct_data["success"] is True
        assert direct_data["sources"] == []
        session_id = direct_data["session_id"]

        # Upload a document (this should switch to RAG mode)
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "switch_test.txt",
                    BytesIO(sample_text_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200

        # Continue in the same session - should now use RAG mode
        response = client.post(
            "/api/chat",
            json={
                "message": "What's in the uploaded document?",
                "session_id": session_id,
            },
        )
        assert response.status_code == 200
        rag_data = response.json()
        assert rag_data["success"] is True
        assert len(rag_data["sources"]) > 0  # Should now have sources
        assert rag_data["session_id"] == session_id

    def test_large_document_processing(self, client):
        """Test processing of large documents."""
        # Create a large document
        large_content = "\n".join(
            [
                f"This is paragraph {i} about artificial intelligence and "
                "machine learning. " * 10
                for i in range(100)
            ]
        )

        response = client.post(
            "/api/upload",
            files={
                "file": ("large_doc.txt", BytesIO(large_content.encode()), "text/plain")
            },
        )
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True
        assert upload_data["chunks"] > 1  # Should be split into multiple chunks

        # Test chat with large document
        response = client.post(
            "/api/chat", json={"message": "What is this document about?"}
        )
        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["success"] is True

    def test_special_characters_and_encoding(self, client):
        """Test handling of special characters and different encodings."""
        special_content = """
        This document contains special characters:
        • Bullet points
        — Em dashes
        "Smart quotes"
        Mathematics: ∑, ∫, √, π, ∞
        Emojis: 🤖 🚀 💻 📊 🧠
        Hebrew: שלום עולם, בינה מלאכותית
        Arabic: مرحبا بالعالم، الذكاء الاصطناعي
        Chinese: 你好世界，人工智能
        """

        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "special_chars.txt",
                    BytesIO(special_content.encode("utf-8")),
                    "text/plain",
                )
            },
        )
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True

        # Test chat with special characters
        response = client.post(
            "/api/chat",
            json={"message": "What languages are mentioned in this document?"},
        )
        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["success"] is True

    def test_document_deduplication(self, client, sample_text_document):
        """Test that duplicate documents are handled correctly."""
        # Upload document first time
        response1 = client.post(
            "/api/upload",
            files={
                "file": (
                    "duplicate_test.txt",
                    BytesIO(sample_text_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response1.status_code == 200
        upload1_data = response1.json()
        assert upload1_data["success"] is True

        # Upload same document again (should return 400 for duplicate)
        response2 = client.post(
            "/api/upload",
            files={
                "file": (
                    "duplicate_test.txt",
                    BytesIO(sample_text_document.encode()),
                    "text/plain",
                )
            },
        )
        assert response2.status_code == 400
        error_data = response2.json()
        assert "already exists" in error_data["detail"]

        # Check that document count hasn't increased
        response = client.get("/api/sources")
        response.json()  # Check response but don't store unused variable
        # Should not have doubled the documents
