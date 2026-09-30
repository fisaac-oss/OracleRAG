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

The project is a rules arbiter for TTRPG players and Game Masters. The core value proposition of the application is to read uploaded PDF rulebooks via Docling, vectorize the content using ChromaDB, and use LangChain to bridge these searches with the local model (`qwen3.5:4b` via Ollama). By employing strict system prompts, we guarantee that the system will output answers only using the provided rulebook context, mitigating the risk of hallucinated rules. The system also includes Short-Term Conversational Memory to handle follow-up questions effectively.

## Main user workflow

1. **User Input:** The user uploads the rulebook and submits a query via a Streamlit user interface.
2. **Processing & Guardrails:** The application service layer (`src/services/ai_service.py`) validates and formats the request. LangChain and ChromaDB identify the relevant context based on similarity search.
3. **Model Response:** The model client calls the `qwen3.5:4b` model via the local Ollama server and returns the formatted response back through the service layer to the UI.

## Architecture

Below is the initial starter architecture. As the project evolves with additional capabilities, the updated diagram will be found at docs/architecture.md.

```text
User
  ↓
Streamlit UI
  ↓
Application / AI Service (src/services/ai_service.py)
  ↓
Vector Database (ChromaDB) / Tools (LangChain & Docling)
  ↓
Model Client (src/models/model_client.py)
  ↓
Ollama (Local LLM Server) - qwen3.5:4b
```
> **Core Architectural Rule:** The user interface must NEVER communicate directly with the model client or Ollama. All interactions must pass through the service layer (`ai_service.py`).

## Model

- **Model used:** `qwen3.5:4b` (running locally via Ollama).
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

- The implementation of **RAG** is the core mechanism of the tool, allowing the application to ingest and perform similarity searches within the original rulebooks uploaded by the users.
- The addition of **Memory / Persistent state** provides conversational functionality, allowing players to ask follow-up or clarifying questions without needing to repeat the full context.

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate oracle-rag
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
MODEL_NAME=qwen3.5:4b
```

### 4. Start Ollama

Make sure Ollama is installed and running locally, then pull your configured model:

```bash
ollama run qwen3.5:4b
```

### 5. Run the application

Run the application from the root directory of the project:

```bash
python -m app.main
```

Then open your browser at `http://localhost:7860`.

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
