#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install youtube-transcript-api')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install langchain')
get_ipython().system('pip install langchain-community')
get_ipython().system('pip install langchain-text-splitters')
get_ipython().system('pip install langgraph')
get_ipython().system('pip install ollama')
get_ipython().system('pip install streamlit')
get_ipython().system('pip install tavily-python')
get_ipython().system('pip install requests')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install python-dotenv')


# In[4]:


print("YouTube Agent project started!")


# In[5]:


import langgraph
import langchain
import faiss
import sentence_transformers
import streamlit
import requests

print("All major libraries are working!")


# In[15]:


youtube_url = "https://www.youtube.com/watch?v=T-D1OfcDW1M"

print(youtube_url)


# In[16]:


from urllib.parse import urlparse, parse_qs

video_id = parse_qs(urlparse(youtube_url).query)["v"][0]

print("Video ID:", video_id)


# In[8]:


from youtube_transcript_api import YouTubeTranscriptApi


# In[17]:


ytt_api = YouTubeTranscriptApi()

transcript = ytt_api.fetch('T-D1OfcDW1M')

print(transcript)


# In[18]:


transcript_text = " ".join(
    snippet.text for snippet in transcript
)

print(transcript_text[:2000])


# In[19]:


print("Number of transcript snippets:", len(transcript))
print("Total characters:", len(transcript_text))


# In[20]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(transcript_text)

print("Number of chunks:", len(chunks))


# In[21]:


for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)


# In[22]:


from sentence_transformers import SentenceTransformer


# In[23]:


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# In[24]:


embeddings = embedding_model.encode(chunks)

print("Number of embeddings:", len(embeddings))
print("Embedding size:", embeddings.shape)  


# In[25]:


import faiss
embedding_dimension = embeddings.shape[1]

print("Embedding dimension:", embedding_dimension)


# In[26]:


index = faiss.IndexFlatL2(embedding_dimension)

print("FAISS index created!")
index.add(embeddings)

print("Number of vectors in FAISS:", index.ntotal)


# In[27]:


query = "What is this video about?"
query_embedding = embedding_model.encode([query])
distances, indices = index.search(query_embedding, k=3)

print("Distances:", distances)
print("Indices:", indices)


# In[28]:


for i in indices[0]:
    print("\n--- Retrieved Chunk ---")
    print(chunks[i])


# In[29]:


import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "What is Retrieval Augmented Generation? Explain simply."
        }
    ]
)

print(response["message"]["content"])


# In[30]:


def retrieve_context(query, k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    return retrieved_chunks


# In[31]:


query = "What does retrieval augmented generation mean?"

retrieved_chunks = retrieve_context(query)

for i, chunk in enumerate(retrieved_chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)


# In[32]:


context = "\n\n".join(retrieved_chunks)

prompt = f"""
Answer the user's question using only the information provided in the context.

If the answer is not available in the context, say:
"I could not find this information in the YouTube video."

Context:
{context}

User Question:
{query}
"""


# In[33]:


response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

print(answer)


# In[34]:


question_1 = "What is retrieval augmented generation?"

query_embedding = embedding_model.encode([question_1])

distances, indices = index.search(query_embedding, k=3)

print("Question:", question_1)
print("Distances:", distances[0])


# In[35]:


question_2 = "What is the capital of France?"

query_embedding = embedding_model.encode([question_2])

distances, indices = index.search(query_embedding, k=3)

print("Question:", question_2)
print("Distances:", distances[0])


# In[36]:


def check_youtube_relevance(query, threshold=1.5):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(query_embedding, k=3)

    best_distance = distances[0][0]

    if best_distance <= threshold:
        return True, best_distance
    else:
        return False, best_distance


# In[37]:


query = "What is retrieval augmented generation?"

is_relevant, distance = check_youtube_relevance(query)

print("Distance:", distance)
print("YouTube relevant:", is_relevant)


# In[38]:


query = "What is the capital of France?"

is_relevant, distance = check_youtube_relevance(query)

print("Distance:", distance)
print("YouTube relevant:", is_relevant)


# In[39]:


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


# In[40]:


query = "What is retrieval augmented generation?"

retrieved_chunks = retrieve_context(query)

result = check_context_with_llm(query, retrieved_chunks)

print("Can YouTube answer?", result)


# In[41]:


query = "What is the capital of France?"

retrieved_chunks = retrieve_context(query)

result = check_context_with_llm(query, retrieved_chunks)

print("Can YouTube answer?", result)


# In[42]:


get_ipython().run_line_magic('pip', 'install tavily-python')


# In[43]:


from tavily import TavilyClient


# In[44]:


from dotenv import load_dotenv
import os

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

print("API key loaded:", tavily_api_key is not None)


# In[45]:


tavily_client = TavilyClient(api_key=tavily_api_key)

print("Tavily client created!")


# In[46]:


query = "What is Retrieval Augmented Generation?"

results = tavily_client.search(
    query=query,
    max_results=3
)

print(results)


# In[47]:


for result in results["results"]:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"][:500])
    print("-" * 50)


# In[49]:


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


# In[50]:


query = "What is Retrieval Augmented Generation?"

results = web_search(query)

for result in results:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"][:300])
    print("-" * 50)


# In[51]:


def route_query(query):

    # Retrieve YouTube context
    retrieved_chunks = retrieve_context(query)

    # Check whether YouTube is relevant
    is_relevant = check_context_with_llm(
        query,
        retrieved_chunks
    )

    if is_relevant:
        return "youtube", retrieved_chunks

    else:
        web_results = web_search(query)
        return "web", web_results


# In[52]:


query = "What is Retrieval Augmented Generation?"

source, results = route_query(query)

print("Source:", source)
print("Results:")
print(results)


# In[54]:


query = "What is the capital of France?"

source, results = route_query(query)

print("Source:", source)
print("Results:")
print(results)


# In[55]:


from typing import TypedDict

class AgentState(TypedDict):
    query: str
    source: str
    context: list
    answer: str


# In[ ]:


def youtube_node(state):

    query = state["query"]

    # Step 1: similarity check
    is_similar, distance = check_youtube_relevance(query)

    if not is_similar:
        return {
            "source": "web",
            "context": []
        }

    # Step 2: LLM verification
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


# In[57]:


state = {
    "query": "What is Retrieval Augmented Generation?",
    "source": "",
    "context": [],
    "answer": ""
}

result = youtube_node(state)

print(result)


# In[58]:


def web_node(state):

    query = state["query"]

    web_results = web_search(query)

    return {
        "source": "web",
        "context": web_results
    }


# In[59]:


state = {
    "query": "What is the capital of France?",
    "source": "",
    "context": [],
    "answer": ""
}

result = web_node(state)

print(result["source"])
print(result["context"])


# In[60]:


def route_node(state):

    if state["source"] == "youtube":
        return "youtube"

    else:
        return "web"


# In[61]:


from langgraph.graph import StateGraph, END


# In[62]:


graph = StateGraph(AgentState)

graph.add_node("youtube", youtube_node)
graph.add_node("web", web_node)

graph.set_entry_point("youtube")

graph.add_conditional_edges(
    "youtube",
    route_node,
    {
        "youtube": END,
        "web": "web"
    }
)

graph.add_edge("web", END)


# In[63]:


agent = graph.compile()

print("Agent created successfully!")


# In[64]:


state = {
    "query": "What is Retrieval Augmented Generation?",
    "source": "",
    "context": [],
    "answer": ""
}

result = agent.invoke(state)

print("Source:", result["source"])
print("Context:")
print(result["context"])


# In[65]:


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


# In[66]:


graph.add_node("answer", answer_node)


# In[67]:


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

print("Final agent created!")


# In[68]:


state = {
    "query": "What is Retrieval Augmented Generation?",
    "source": "",
    "context": [],
    "answer": ""
}

result = agent.invoke(state)

print("Source:", result["source"])
print("\nFinal Answer:")
print(result["answer"])


# In[75]:


def ask_agent(query):

    state = {
        "query": query,
        "source": "",
        "context": [],
        "answer": ""
    }

    result = agent.invoke(state)

    print("Source:", result["source"])
    print("\nAnswer:")
    print(result["answer"])


# In[ ]:




