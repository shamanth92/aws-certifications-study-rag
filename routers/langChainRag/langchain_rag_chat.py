import uuid
from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from services.langchain_rag_services.rag_retrieval_service import generate_answer
from services.conversation_service import get_history

router = APIRouter(prefix="/langchain/chat/rag", tags=["langchain-rag-chat"])


# Single endpoint for both features of the LangChain pipeline: pass
# mode="qa" (default) to answer a question, or mode="exam" to generate
# practice exam questions on a topic. Same retrieval step either way --
# only the prompt used inside generate_answer() changes based on mode.
@router.post("/", response_model=ChatResponse)
async def rag_chat_endpoint(request: ChatRequest):
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        answer = await generate_answer(request.message, request.mode, conversation_id)
        return ChatResponse(answer=answer, conversation_id=conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lets the frontend restore a conversation's message history, e.g. after a
# page refresh, so the visible chat thread doesn't just disappear.
@router.get("/{conversation_id}")
def get_conversation_history(conversation_id: str):
    try:
        return {"messages": get_history(conversation_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
