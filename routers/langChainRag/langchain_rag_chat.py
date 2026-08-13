import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest, ChatResponse
from services.langchain_rag_services.rag_retrieval_service import stream_answer
from services.conversation_service import get_history

router = APIRouter(prefix="/langchain/chat/rag", tags=["langchain-rag-chat"])


# Single endpoint for both features of the LangChain pipeline: pass
# mode="qa" (default) to answer a question, or mode="exam" to generate
# practice exam questions on a topic. Same retrieval step either way --
# only the prompt used inside stream_answer() changes based on mode.
#
# Streams the answer as plain text chunks as they're generated, rather than
# waiting for the full response -- the client reads the response body as a
# stream (e.g. via fetch()'s ReadableStream). conversation_id is returned via
# a response header instead of the JSON body, since the body itself is now
# the raw streamed answer text, not a JSON object.
@router.post("/")
async def rag_chat_endpoint(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())

    return StreamingResponse(
        stream_answer(request.message, request.mode, conversation_id),
        media_type="text/plain",
        headers={"X-Conversation-Id": conversation_id}
    )


# Lets the frontend restore a conversation's message history, e.g. after a
# page refresh, so the visible chat thread doesn't just disappear.
@router.get("/{conversation_id}")
def get_conversation_history(conversation_id: str):
    try:
        return {"messages": get_history(conversation_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
