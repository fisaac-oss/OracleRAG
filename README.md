# Oracle RAG

## Team members

- Isaac Lívi dos Santos (isaac.lividossantos@student.hamk.fi)
- Ronak Kadyan (ronak.kadyan@student.hamk.fi)
- Shreshtha Kumar (shreshtha.kumar@student.hamk.fi)

## Problem

### Intended users
- The primary target users for this application are players and Game Masters of Tabletop Role-Playing Games (TTRPG).

### Problem statement
- Players and Game Masters need to be able to answer questions regarding game rules quickly during a session.
- The application solves this problem by allowing users to upload their own rulebooks and extract accurate answers from them.

### Why AI is appropriate
- Traditional deterministic solutions rely on literal keyword searches, whereas TTRPG rules require natural language understanding to parse complex context and intent. AI (specifically through RAG) perfectly fills the need to both find the rule and interpret it to provide a direct answer to the user's question.

## Solution

The project is a rules arbiter for TTRPG players and Game Masters. The core value proposition of the application is to read uploaded PDF rulebooks with Docling, store their embeddings in ChromaDB, and retrieve relevant passages through a native Python RAG pipeline with BM25 lexical re-ranking. The retrieved context is passed to a local Ollama model (`llama3.2:3b` by default). By employing strict system prompts, we guarantee that the system will output answers only using the provided rulebook context, mitigating the risk of hallucinated rules. The system also includes short-term conversational memory to handle follow-up questions effectively.

## Main user workflow

1. **User Input:** The user uploads the rulebook and submits a query via a Streamlit user interface.
2. **Processing & Guardrails:** The application service layer (`src/services/ai_service.py`) validates and formats the request. The native RAG pipeline combines ChromaDB similarity search with BM25 reciprocal-rank-fusion re-ranking to identify relevant context.
3. **Model Response:** The model client calls the configured model via the local Ollama server and returns the formatted response back through the service layer to the UI.

## Architecture

The implemented architecture is documented in [`docs/architecture.md`](docs/architecture.md).

```text
User
  ↓
Streamlit UI
  ↓
Application / AI Service (src/services/ai_service.py)
  ↓
Document Processing (Docling)
  ↓
Hybrid Retrieval (ChromaDB + BM25 RRF)
  ↓
Model Client (src/models/model_client.py)
  ↓
Ollama (Local LLM Server) - configured model
```
> **Core Architectural Rule:** The user interface NEVER communicate directly with the model client or Ollama. All interactions pass through the service layer (`ai_service.py`).

## Model

- **Model used:** `llama3.2:3b` by default, configurable through `MODEL_NAME` (running locally via Ollama).
- **Selection rationale:** This model was chosen because it is lightweight and performs efficiently on local hardware.

## Additional AI capability

Select at least one additional capability to implement for your final project:

- [x] RAG (Retrieval-Augmented Generation)
- [ ] Tools / External API integration
- [ ] Model Context Protocol (MCP)
- [ ] Agentic workflow (Model-selected actions based on observations)
- [x] Memory / Persistent state
- [ ] Multimodal interaction (Text + Images)
- [ ] Other: ______________________

### Capability justification

- The implementation of **RAG** is the core mechanism of the tool, allowing the application to ingest and search original rulebooks uploaded by users.
- The addition of **Memory / Persistent state** provides conversational functionality, allowing players to ask follow-up or clarifying questions without repeating the full context.

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

The environment is named `oracle_rag` in [`environment.yml`](environment.yml):

```bash
conda activate oracle_rag
```

### 3. Configure environment variables

Copy `.env.example` to create your local `.env` configuration file:

On Linux / macOS:
```bash
cp .env.example .env
```

On Windows (Command Prompt / PowerShell):
```powershell
copy .env.example .env
```

Ensure `.env` contains valid values for `OLLAMA_BASE_URL` and `MODEL_NAME`:
```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.2:3b
```

### 4. Start Ollama

Make sure Ollama is installed and running locally, then pull your configured model:

```bash
ollama run llama3.2:3b
```

### 5. Run the application

From the project root, start the Streamlit application with:

```bash
streamlit run app/main.py
```

Streamlit serves the application at `http://localhost:8501` by default. The
application requires Streamlit's runtime.

### 6. Run automated tests

```bash
pytest
```

## Evaluation

Describe your evaluation methodology and summarize key results. Starter test cases can be found in [`evaluation/test_cases.json`](evaluation/test_cases.json).

Refer to [`evaluation/README.md`](evaluation/README.md) for guidelines on defining success, edge cases, and failure scenarios.

## Known limitations

- **Memory Restraints:** Conversational memory will be strictly limited to the last few interactions. While this limits the ability to recall events from early in the session, it is intentional to avoid polluting the prompt context and degrading model performance.
- **Fallback Mechanism:** The model will not attempt to guess or extrapolate rules. If the similarity search returns a low confidence score, or if the model cannot find the answer in the retrieved text, the system will output a direct refusal message rather than guessing.

## Future improvements

- List planned feature enhancements, architectural refactorings, or future capabilities.
