import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from services.conversation_service import get_history, add_message

load_dotenv()

# Module-level (created once, reused across requests) since these don't hold
# any state tied to a specific database snapshot.
# Embeddings stay on OpenAI -- Anthropic has no embeddings API, so retrieval
# is unaffected by the switch to Claude for chat/generation below.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# CLAUDE_API_KEY is passed explicitly since ChatAnthropic defaults to reading
# ANTHROPIC_API_KEY from the environment, not CLAUDE_API_KEY.
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("CLAUDE_API_KEY"))

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

Each question must test understanding of the underlying AWS AI/ML concept -- e.g. how a service \
works, when to use it over an alternative, how it fits a given scenario, or what a term/feature \
means. This should read exactly like a real AIF-C01 exam question.

Never write a question about the source material itself -- do not ask which document, whitepaper, \
guide, or exam guide mentions/lists/includes something, and do not make the document's title, \
metadata, or structure the subject of the question or an answer option. The context is only a \
source of facts to build real scenario questions from, not a topic to be quizzed on.

Do not reference the source material anywhere inside the question text either -- never write \
phrases like "according to the guide", "based on the provided context", "per the whitepaper", or \
similar. State the scenario directly, exactly as a real exam question would, with no indication \
that the question was derived from retrieved material. Save any reference to where the fact came \
from for the Citation field only.

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
# at module level to ensure fresh database connections and avoid stale state.
#
# This is a generator (not a plain async function) so the router can stream
# tokens to the client as they arrive, instead of waiting for the full answer.
# The rewrite step still runs as a single non-streamed call first, since it's
# short and the retriever needs the final standalone question before it can
# even start searching -- only the final answer generation is streamed.
async def stream_answer(question: str, mode: str = "qa", conversation_id: str | None = None):
    prompt = exam_prompt if mode == "exam" else qa_prompt  # mode value is the enum's string value
    database_url = os.getenv("DATABASE_URL")
    # conn = psycopg.connect(database_url)
    vector_store = PGVector.from_existing_index(
        embedding=embeddings,
        connection=database_url,
        collection_name="aws_rag_documents",
        async_mode=True
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

    answer_chunks = []
    async for chunk in chain.astream(retrieval_question):
        answer_chunks.append(chunk)
        yield chunk

    # Only persist to history once the full answer has streamed successfully --
    # a connection drop mid-stream shouldn't leave a user message with no reply.
    if conversation_id:
        full_answer = "".join(answer_chunks)
        add_message(conversation_id, "user", question)
        add_message(conversation_id, "assistant", full_answer)

async def rewrite_question(question: str, prior_messages: list[dict]) -> str:
    if not prior_messages:
        return question

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in prior_messages)
    chain = rewrite_prompt | llm | StrOutputParser()
    return await chain.ainvoke({"history": history_text, "question": question})

