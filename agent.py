import os
import faiss
import ollama

from dotenv import load_dotenv
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict
from youtube_transcript_api import YouTubeTranscriptApi


# -----------------------------
# 1. Load environment
# -----------------------------

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


# -----------------------------
# 2. YouTube transcript
# -----------------------------

video_id = "T-D1OfcDW1M"

ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)

transcript_text = " ".join(
    snippet.text for snippet in transcript
)


# -----------------------------
# 3. Create chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(transcript_text)


# -----------------------------
# 4. Create embeddings
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)


# -----------------------------
# 5. Create FAISS index
# -----------------------------

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dimension)

index.add(embeddings)


# -----------------------------
# 6. Retrieve YouTube context
# -----------------------------

def retrieve_context(query, k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        query_embedding,
        k
    )

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    return retrieved_chunks


# -----------------------------
# 7. LLM relevance check
# -----------------------------

def check_context_with_llm(query, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are checking whether the provided context can answer the question.

Context:
{context}

Question:
{query}

Can the context answer the question?

Reply with only:
YES
or
NO
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"].strip().upper()

    if "YES" in result:
        return True
    else:
        return False


# -----------------------------
# 8. Web search
# -----------------------------

def web_search(query, max_results=3):

    results = tavily_client.search(
        query=query,
        max_results=max_results
    )

    web_results = []

    for result in results["results"]:

        web_results.append({
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        })

    return web_results


# -----------------------------
# 9. Agent State
# -----------------------------

class AgentState(TypedDict):

    query: str
    source: str
    context: list
    answer: str


# -----------------------------
# 10. YouTube Node
# -----------------------------

def youtube_node(state):

    query = state["query"]

    retrieved_chunks = retrieve_context(query)

    is_relevant = check_context_with_llm(
        query,
        retrieved_chunks
    )

    if is_relevant:

        return {
            "source": "youtube",
            "context": retrieved_chunks
        }

    else:

        return {
            "source": "web",
            "context": []
        }


# -----------------------------
# 11. Web Node
# -----------------------------

def web_node(state):

    query = state["query"]

    web_results = web_search(query)

    return {
        "source": "web",
        "context": web_results
    }


# -----------------------------
# 12. Routing
# -----------------------------

def route_node(state):

    if state["source"] == "youtube":
        return "youtube"

    else:
        return "web"


# -----------------------------
# 13. Answer Node
# -----------------------------

def answer_node(state):

    query = state["query"]
    context = state["context"]
    source = state["source"]

    if source == "youtube":

        context_text = "\n\n".join(context)

    else:

        context_text = "\n\n".join(
            [item["content"] for item in context]
        )

    prompt = f"""
Answer the user's question using only the provided context.

Context:
{context_text}

Question:
{query}

Give a clear and simple answer.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "answer": response["message"]["content"]
    }


# -----------------------------
# 14. Build LangGraph
# -----------------------------

graph = StateGraph(AgentState)

graph.add_node("youtube", youtube_node)
graph.add_node("web", web_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("youtube")

graph.add_conditional_edges(
    "youtube",
    route_node,
    {
        "youtube": "answer",
        "web": "web"
    }
)

graph.add_edge("web", "answer")
graph.add_edge("answer", END)

agent = graph.compile()