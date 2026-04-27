from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@router.post("")
async def chat(req: ChatRequest):
    return {"answer": "Set OPENAI_API_KEY in .env to enable chatbot."}
