<img width="1281" height="740" alt="web search" src="https://github.com/user-attachments/assets/36785714-4d6e-4c17-b0ec-d85762c7df53" />
# 🎥 YouTube Research Assistant

> An Agentic AI research assistant that answers questions from a YouTube video using RAG and automatically falls back to web search when the required information is not available in the video.

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20Workflow-1C3C3C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />

</p>

---

## 🚀 Overview

The **YouTube Research Assistant** is an Agentic AI application that allows users to ask questions about a YouTube video.

The system first searches the video's transcript using a **Retrieval-Augmented Generation (RAG)** pipeline.

If the retrieved information is not sufficient to answer the question, the agent automatically performs a **web search using Tavily** and uses the retrieved web information to generate the answer.

The final response is generated locally using **Qwen3 through Ollama**.

---

## ✨ Key Features

- 🎥 Extracts transcripts from YouTube videos
- ✂️ Splits transcripts into manageable chunks
- 🧠 Generates semantic embeddings using Sentence Transformers
- 🔎 Performs vector similarity search using FAISS
- 🤖 Uses an LLM to evaluate whether retrieved context can answer the question
- 🔀 Uses LangGraph for agentic routing
- 🌐 Falls back to Tavily web search when YouTube context is insufficient
- 🏠 Runs the LLM locally using Ollama
- 💬 Provides an interactive Streamlit interface
- 🔐 Keeps API credentials in `.env`

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │     User Question   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     LangGraph       │
                         │   Agent Workflow    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   YouTube RAG       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Transcript Chunks   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ SentenceTransformer│
                         │    Embeddings       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FAISS         │
                         │   Vector Retrieval  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LLM Relevance     │
                         │       Check         │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                    Relevant                 Not Relevant
                       │                         │
                       ▼                         ▼
                ┌─────────────┐          ┌─────────────┐
                │   YouTube   │          │    Tavily   │
                │   Context   │          │ Web Search  │
                └──────┬──────┘          └──────┬──────┘
                       │                         │
                       └────────────┬────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Qwen3         │<img width="1355" height="757" alt="youtube answer" src="https://github.com/user-attachments/assets/74335445-992d-41fa-80fd-b9f48aa56d91" />

                         │   Local LLM/Ollama  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Final Answer     │
                         └─────────────────────┘
