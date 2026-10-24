# Architecture Documentation

This document describes the implemented architecture of Oracle RAG.

The application uses a native Python pipeline with Streamlit as its only UI
runtime.

## Implemented Architecture

```text
User
  ↓
Streamlit UI (app/ui.py)
  ↓
Application / AI Service (src/services/ai_service.py)
  ├── ConversationMemory (src/capabilities/memory.py)
  ├── RAGPipeline (src/capabilities/rag.py)
  │     ├── Docling document conversion and semantic chunking
  │     ├── ChromaDB persistent vector storage
  │     └── BM25 lexical re-ranking with reciprocal rank fusion
  └── OllamaModelClient (src/models/model_client.py)
          ↓
      Ollama (local inference server)
          ↓
      Configured local model (llama3.2:3b by default)
```

### Core Design Rules

1. **Strict UI isolation:** The Streamlit UI delegates model, retrieval, ingestion, and deletion operations to `src/services/ai_service.py`. It does not call Ollama directly.
2. **Service orchestration:** `AIService` validates requests, combines conversation history with retrieved rulebook context, and returns structured response schemas.
3. **Hybrid retrieval:** `RAGPipeline` stores uploaded rulebook chunks in persistent ChromaDB and improves vector candidates with BM25 lexical re-ranking using reciprocal rank fusion.
4. **Conversation state:** `ConversationMemory` keeps a bounded set of recent user/assistant turns per Streamlit chat session.
5. **Controlled failures:** The model and service layers translate connection, model, retrieval, and validation failures into user-understandable structured responses.
6. **Data schemas:** Request inputs and response structures are validated using Pydantic models in `src/schemas/responses.py`.

## Local Runtime

Create and activate the environment declared in [`environment.yml`](../environment.yml):

```bash
conda env create -f environment.yml
conda activate oracle_rag
```

Start the application from the project root:

```bash
streamlit run app/main.py
```

The UI is available at `http://localhost:8501`. `app/main.py` is the
Streamlit entry point and requires Streamlit's runtime.
