from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from services.conversation_service import get_history, add_message

# Module-level (created once, reused across requests) since these don't hold
# any state tied to a specific ChromaDB collection snapshot.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5-mini")

# Two prompt templates share the same retrieval step but produce very
# different output -- this is the "one endpoint, swap the prompt by mode"
# design: qa_prompt answers directly, exam_prompt generates practice questions.
qa_prompt = ChatPromptTemplate.from_template("""
You are an AWS Certified AI Practitioner tutor. Answer the question using only the context below.
If the answer is not in the context, say you don't know.
If the question refers to something ambiguous (e.g. "it", "its", "that", "this") and the topic \
is not clear from the question itself, do not guess -- ask the user to clarify what they mean \
instead of answering.
Provide concise exam-focused explanations. Keep responses under 500 words.
Answer directly and naturally, as a tutor speaking to a student. Do not say phrases like \
"based on the provided context", "according to the document", or similar meta-references to \
where the information came from -- just state the answer.
----------------
{context}
----------------
Question: {question}
""")

exam_prompt = ChatPromptTemplate.from_template("""
You are an AWS Certified AI Practitioner exam coach. Using the context below, generate exam-style \
multiple choice questions on the topic the user asks about.

If the topic refers to something ambiguous (e.g. "it", "its", "that", "this") and the topic is not \
clear from the request itself, do not guess -- ask the user to clarify which topic they want \
questions on instead of generating questions.

Format each question as:
Q: <question>
A) ...
B) ...
C) ...
D) ...
Answer: <correct option>
Explanation: <brief explanation>
Citation: <source from context>

Generate 2 questions. Base the questions only on the context provided, but they should mock actual scenario based questions that you might see in the real exam.
Can you add citations to each question informing the user which part of the context was used to generate the question?
----------------
{context}
----------------
Topic: {question}
""")

rewrite_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and a follow-up question, rewrite the follow-up as a standalone question that includes all necessary context. If the follow-up is already standalone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:""")


# Builds and runs an LCEL chain (LangChain Expression Language -- the `|`
# pipe operator composes runnables into a pipeline):
#   1. {"context": retriever, "question": RunnablePassthrough()}
#      runs the retriever on the input question to fetch relevant chunks,
#      while passing the original question through unchanged, producing
#      {"context": [...chunks], "question": "..."}
#   2. prompt   -> fills the template with that dict
#   3. llm      -> sends the filled prompt to the chat model
#   4. StrOutputParser() -> extracts the plain string answer from the LLM response
#
# vector_store/retriever/chain are (re)built on every call rather than cached
# at module level -- caching them would hold a reference to the ChromaDB
# collection that goes stale after a delete_collection() call (e.g. via the
# DELETE /langchain/ingestion/ endpoint), causing "collection does not exist"
# errors on the next query.
async def generate_answer(question: str, mode: str = "qa", conversation_id: str | None = None) -> str:
    prompt = exam_prompt if mode == "exam" else qa_prompt  # mode value is the enum's string value
    vector_store = Chroma(
        collection_name="aws-rag-documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prior_messages = get_history(conversation_id) if conversation_id else []
    retrieval_question = await rewrite_question(question, prior_messages)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = await chain.ainvoke(retrieval_question)

    if conversation_id:
        add_message(conversation_id, "user", question)
        add_message(conversation_id, "assistant", answer)

    return answer

async def rewrite_question(question: str, prior_messages: list[dict]) -> str:
    if not prior_messages:
        return question

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in prior_messages)
    chain = rewrite_prompt | llm | StrOutputParser()
    return await chain.ainvoke({"history": history_text, "question": question})

