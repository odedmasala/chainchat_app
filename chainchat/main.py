# chainchat_app/chainchat/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

# כאן מגדירים את ה-FastAPI instance בשם app
app = FastAPI(title="ChainChat")

# מודלים של Pydantic לבקשה ותשובה
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# Endpoint בסיסי להדגמה
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    return ChatResponse(reply=f"Echo: {req.message}")
