import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import settings


class ChatService:

    def __init__(self):
        import os

        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embedding_type = "openai"
            print("🔧 Using OpenAI embeddings")
        except Exception as e:
            print(f"⚠️  OpenAI embeddings failed, using local embeddings: {e}")
            try:
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
                print(
                    "🔧 Using multilingual Sentence Transformer embeddings (Hebrew supported)"
                )
            except Exception:
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                print("🔧 Using basic Sentence Transformer embeddings")
            self.embedding_type = "local"

        self.llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)

        self.documents: List[Document] = []
        self.vector_store: Optional[FAISS] = None
        self.document_sources: Dict[str, Dict[str, Any]] = {}

        self.sessions: Dict[str, Dict[str, Any]] = {}

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def add_document(self, text: str, filename: str) -> Dict[str, Any]:
        try:
            print(f"📄 Processing document: {filename}")
            print(f"📊 Text length: {len(text)} characters")

            doc_hash = hashlib.md5(text.encode()).hexdigest()

            if doc_hash in self.document_sources:
                print(f"⚠️  Document already exists: {doc_hash}")
                return {
                    "success": False,
                    "message": "Document already exists in knowledge base",
                    "document_id": doc_hash,
                }

            print(f"✂️  Splitting text into chunks...")
            chunks = self.text_splitter.split_text(text)
            print(f"✅ Created {len(chunks)} chunks")

            print(f"📝 Creating document objects...")
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk_id": i,
                        "document_id": doc_hash,
                        "added_at": datetime.now().isoformat(),
                    },
                )
                documents.append(doc)

            print(f"📚 Adding {len(documents)} documents to collection...")
            self.documents.extend(documents)

            self.document_sources[doc_hash] = {
                "filename": filename,
                "chunks": len(chunks),
                "added_at": datetime.now().isoformat(),
                "character_count": len(text),
            }

            print(f"🔍 Rebuilding vector store...")
            self._rebuild_vector_store()
            print(f"✅ Document processing complete!")

            return {
                "success": True,
                "message": f"Document processed into {len(chunks)} chunks",
                "document_id": doc_hash,
                "chunks": len(chunks),
            }

        except Exception as e:
            print(f"❌ Error in add_document: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "message": f"Error processing document: {str(e)}"}

    def ask(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not session_id:
                session_id = str(uuid.uuid4())

            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "memory": None,
                    "created_at": datetime.now().isoformat(),
                    "message_count": 0,
                }

            session = self.sessions[session_id]

            if not self.vector_store:

                if session["memory"] is None or hasattr(
                    session["memory"], "output_key"
                ):
                    direct_memory = ConversationBufferWindowMemory(
                        memory_key="history", return_messages=True, k=5
                    )

                    if (
                        session["memory"]
                        and hasattr(session["memory"], "chat_memory")
                        and hasattr(session["memory"].chat_memory, "messages")
                    ):
                        direct_memory.chat_memory.messages = session[
                            "memory"
                        ].chat_memory.messages.copy()

                    session["memory"] = direct_memory

                conversation_chain = ConversationChain(
                    llm=self.llm, memory=session["memory"], verbose=False
                )

                response = conversation_chain.predict(input=question)

                session["message_count"] += 1
                session["last_activity"] = datetime.now().isoformat()

                return {
                    "success": True,
                    "answer": response,
                    "sources": [],
                    "session_id": session_id,
                    "message_count": session["message_count"],
                    "mode": "direct_chat",
                }
            else:
                if (
                    session["memory"] is None
                    or not hasattr(session["memory"], "output_key")
                    or session["memory"].output_key != "answer"
                ):
                    rag_memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer",
                        k=5,
                    )

                    if (
                        session["memory"]
                        and hasattr(session["memory"], "chat_memory")
                        and hasattr(session["memory"].chat_memory, "messages")
                    ):
                        rag_memory.chat_memory.messages = session[
                            "memory"
                        ].chat_memory.messages.copy()

                    session["memory"] = rag_memory

                from langchain.prompts import PromptTemplate

                multilingual_prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""You are a helpful AI assistant that can communicate in multiple languages including Hebrew, English, Arabic, and others. Use the following pieces of context to answer the question at the end. 

Important instructions:
1. If the question is in Hebrew (עברית), respond in Hebrew unless specifically asked otherwise
2. If the question is in English, respond in English unless asked otherwise  
3. If the question refers to "the file", "the document", "הקובץ", "המסמך", it refers to the uploaded document(s)
4. Maintain conversation context across different languages
5. When switching languages, acknowledge the previous conversation context

Context from documents:
{context}

Question: {question}

Answer in the same language as the question, and provide helpful, accurate information based on the context.""",
                )

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 6}
                    ),
                    memory=session["memory"],
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": multilingual_prompt},
                )

                enhanced_question = question

                if any(
                    word in question for word in ["הקובץ", "המסמך", "הטקסט", "המידע"]
                ):
                    enhanced_question = f"בהתבסס על המסמך שהועלה, {question}"
                elif "תסביר" in question and (
                    "בעברית" in question or len(question.split()) <= 3
                ):
                    enhanced_question = f"תסביר את התוכן של המסמך שהועלה בעברית"
                elif question.strip() in [
                    "תסביר על הקובץ בבקשה",
                    "תן לי סיכום",
                    "מה יש במסמך",
                ]:
                    enhanced_question = f"בהתבסס על המסמך שהועלה, {question}"
                elif any(
                    word in question.lower()
                    for word in ["the file", "the document", "this document"]
                ):
                    enhanced_question = f"Based on the uploaded document, {question}"
                elif question.lower().strip() in [
                    "explain in english",
                    "summarize this",
                    "what's in this",
                ]:
                    enhanced_question = f"Based on the uploaded document, {question}"

                result = qa_chain.invoke({"question": enhanced_question})

                sources = []
                for doc in result.get("source_documents", []):
                    source_info = {
                        "filename": doc.metadata.get("source", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "content_preview": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                    }
                    if source_info not in sources:
                        sources.append(source_info)

                session["message_count"] += 1
                session["last_activity"] = datetime.now().isoformat()

                return {
                    "success": True,
                    "answer": result["answer"],
                    "sources": sources,
                    "session_id": session_id,
                    "message_count": session["message_count"],
                    "mode": "rag_chat",
                }

        except Exception as e:
            error_str = str(e)
            print(f"❌ Error in ask method: {error_str}")

            if (
                "quota" in error_str.lower()
                or "429" in error_str
                or "insufficient_quota" in error_str.lower()
            ):
                return {
                    "success": False,
                    "message": "OpenAI API quota exceeded",
                    "answer": """🚫 **OpenAI API Quota Exceeded**

I can process your documents perfectly (using local embeddings), but I need OpenAI API access to generate chat responses.

**💡 Solutions:**
1. **Add credits** to your OpenAI account at https://platform.openai.com/billing
2. **Wait for quota reset** (if on monthly plan)  
3. **Check your billing details** at https://platform.openai.com/settings/billing

**📊 Current Status:**
✅ Document processing: Working (local embeddings)
❌ Chat responses: Blocked (needs OpenAI API)

Your documents are ready - I just need API access to answer questions about them!""",
                }

            return {
                "success": False,
                "message": f"Error processing question: {error_str}",
                "answer": "I encountered an error while processing your question. Please try again.",
            }

    def get_sources(self) -> Dict[str, Any]:
        return {
            "documents": self.document_sources,
            "total_documents": len(self.document_sources),
            "total_chunks": len(self.documents),
        }

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"success": False, "message": "Session not found"}

        session = self.sessions[session_id]
        memory = session["memory"]

        messages = []
        if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
            for message in memory.chat_memory.messages:
                messages.append(
                    {
                        "type": message.__class__.__name__.lower(),
                        "content": message.content,
                    }
                )

        return {
            "success": True,
            "session_id": session_id,
            "created_at": session["created_at"],
            "message_count": session["message_count"],
            "messages": messages,
        }

    def _rebuild_vector_store(self):
        if self.documents:
            try:
                self.vector_store = FAISS.from_documents(
                    self.documents, self.embeddings
                )
                print(f"✅ Vector store created with {self.embedding_type} embeddings")
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    print(f"💳 OpenAI quota exceeded, switching to local embeddings...")
                    print(
                        f"📥 Downloading sentence-transformers model (first time only)..."
                    )
                    print(f"⏳ This may take 1-3 minutes - please wait...")

                    try:
                        self.embeddings = SentenceTransformerEmbeddings(
                            model_name="paraphrase-multilingual-MiniLM-L12-v2"
                        )
                        print(
                            f"✅ Multilingual model downloaded and loaded successfully!"
                        )
                    except Exception:
                        self.embeddings = SentenceTransformerEmbeddings(
                            model_name="all-MiniLM-L6-v2"
                        )
                        print(f"✅ Basic model downloaded and loaded successfully!")
                    self.embedding_type = "local"

                    print(f"🔄 Creating vector store with local embeddings...")
                    self.vector_store = FAISS.from_documents(
                        self.documents, self.embeddings
                    )
                    print(f"✅ Vector store created with local embeddings")
                else:
                    raise e


chat_service = ChatService()
