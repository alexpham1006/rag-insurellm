from chromadb import PersistentClient
from litellm import completion
from openai import OpenAI
from tenacity import retry, stop_after_attempt

from rag_insurellm.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FINAL_K,
    RETRIEVAL_K,
    MODEL,
    WAIT,
)
from rag_insurellm.models import Result
from rag_insurellm.rerank import rerank


openai = OpenAI()
chroma = PersistentClient(path=str(CHROMA_PATH))
collection = chroma.get_or_create_collection(COLLECTION_NAME)


@retry(wait=WAIT, stop=stop_after_attempt(3), reraise=True)
def rewrite_query(question, history=None):
    if history is None:
        history = []

    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, extra):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    query = openai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question, history=None):
    rewritten_question = rewrite_query(original_question, history)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]
