from dotenv import load_dotenv
load_dotenv()

import os
import io
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available. Using pypdf only for PDF extraction.")

from .config import settings
from .chat import chat_service

def extract_pdf_text(content: bytes) -> str:
    
    def try_pypdf_extraction(content: bytes) -> str:
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        
        text_content = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n\n"
        
        return text_content.strip()
    
    def try_pymupdf_extraction(content: bytes) -> str:
        pdf_file = io.BytesIO(content)
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        
        text_content = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text_content += page_text + "\n\n"
        
        doc.close()
        return text_content.strip()
    
    try:
        text_content = try_pypdf_extraction(content)
        if text_content:
            return text_content
    except Exception as e:
        print(f"pypdf extraction failed: {e}")
    
    if PYMUPDF_AVAILABLE:
        try:
            text_content = try_pymupdf_extraction(content)
            if text_content:
                return text_content
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
    
    error_msg = "Failed to extract text from PDF"
    if PYMUPDF_AVAILABLE:
        error_msg += " using both pypdf and PyMuPDF"
    else:
        error_msg += " using pypdf"
    
    error_msg += (
        ". The PDF appears to be:\n\n"
        "🔍 **Most likely: Image-based/scanned PDF**\n"
        "   - Common with academic lecture notes\n"
        "   - Hebrew text from scanned documents\n"
        "   - Requires OCR (Optical Character Recognition)\n\n"
        "📝 **Solutions:**\n"
        "   1. Convert to searchable PDF using Adobe Acrobat\n"
        "   2. Use OCR tools like Tesseract with Hebrew support\n"
        "   3. Try Google Drive (Upload → Right-click → Open with Google Docs)\n"
        "   4. Use online OCR services for Hebrew text\n\n"
        "🔒 **Other possibilities:**\n"
        "   • Password protected/encrypted\n"
        "   • Corrupted file format\n"
        "   • Contains only images without text layer"
    )
    
    raise ValueError(error_msg)

app = FastAPI(
    title=settings.app_name,
    description="RAG-powered chat application with document processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: str
    sources: list = []
    session_id: Optional[str] = None
    message_count: Optional[int] = None
    message: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks: Optional[int] = None

@app.get("/")
async def read_root():
    static_index = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    return {"message": f"Welcome to {settings.app_name}! Upload documents and start chatting."}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    result = chat_service.ask(request.message, request.session_id)
    
    if not result["success"]:
        return ChatResponse(
            success=False,
            answer=result.get("answer", ""),
            message=result.get("message", "An error occurred")
        )
    
    return ChatResponse(
        success=True,
        answer=result["answer"],
        sources=result.get("sources", []),
        session_id=result.get("session_id"),
        message_count=result.get("message_count")
    )

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    allowed_types = ['.txt', '.md', '.csv', '.json', '.py', '.js', '.html', '.css', '.pdf']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        if file_ext == '.pdf':
            print(f"Processing PDF file: {file.filename}, size: {len(content)} bytes")
            text_content = extract_pdf_text(content)
            print(f"Extracted text length: {len(text_content)} characters")
            if not text_content.strip():
                raise ValueError("No text content extracted from PDF")
        else:
            text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        print(f"Unicode decode error for file: {file.filename}")
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        print(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    result = chat_service.add_document(text_content, file.filename)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return UploadResponse(
        success=True,
        message=result["message"],
        document_id=result.get("document_id"),
        chunks=result.get("chunks")
    )

@app.get("/api/health")
async def health_check():
    sources_info = chat_service.get_sources()
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "documents_loaded": sources_info["total_documents"],
        "total_chunks": sources_info["total_chunks"]
    }

@app.get("/api/sources")
async def get_sources():
    return chat_service.get_sources()

@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    result = chat_service.get_session_history(session_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "success": False,
        "message": "An unexpected error occurred",
        "detail": str(exc) if settings.app_name == "development" else "Internal server error"
    }
