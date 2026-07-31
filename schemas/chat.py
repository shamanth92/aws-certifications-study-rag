from pydantic import BaseModel
from enum import Enum


# Enum instead of a plain string so FastAPI validates the value automatically
# (invalid modes return a 422 instead of silently falling through) and Swagger
# docs show the allowed values as a dropdown.
class ChatMode(str, Enum):
    qa = "qa"      # answer the user's question directly from retrieved context
    exam = "exam"  # generate exam-style practice questions on the given topic


class ChatRequest(BaseModel):
    message: str
    mode: ChatMode = ChatMode.qa
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    # Optional because the RAG chat endpoints are stateless (no multi-turn
    # conversation tracking yet) -- only the plain chat feature would set this.
    conversation_id: str | None = None
    answer: str
