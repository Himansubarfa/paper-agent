[README.md](https://github.com/user-attachments/files/34814549/README.md)
# 📄 Paper-to-Presentation & Viva Agent

An academic AI assistant that ingests a research paper (PDF) and automatically produces:

1. A **sectioned summary** grounded in the paper.
2. A structured extraction of **contributions, methods, and results**.
3. A downloadable **PowerPoint (.pptx) slide deck**.
4. Likely **viva/examiner questions** with model answers, interactive scoring, and feedback.
5. **Free-form Q&A** with page-level citations.

Built with **Streamlit**, **LangChain (LCEL)**, and **ChromaDB**. It supports fast API inference (Groq, OpenAI, Anthropic) or 100% offline local inference (Ollama).

![Paper Agent UI Placeholder](https://via.placeholder.com/1200x600.png?text=Paper+Agent+UI+Screenshot)

---

## ✨ Features

- **Summarizer Agent**: Uses map-reduce over document sections to generate a comprehensive, section-by-section summary.
- **Concept Extractor**: Uses targeted RAG to find the problem, contributions, methods, datasets, metrics, and limitations, with strict chunk-ID citation validation.
- **Slide Generator**: Converts the summary and concepts into a structured JSON schema, then renders a native 16:9 `.pptx` file using `python-pptx`.
- **Interactive Viva Agent**: Generates recall, application, and critical questions. A full state machine lets you answer, receive rubric scores (1–5), get follow-up questions, and generate a final study report.
- **Agentic Router**: Detects your intent from your chat input or UI selection, and keeps the selected task until you clearly switch.

---

## 🚀 Quick Start (API Mode)

*Fastest setup. Recommended for most users.*

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/paper-agent.git
   cd paper-agent
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure the environment**

   Copy the example file:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and set your preferred provider and API key:

   ```env
   # Example: using Groq (fast, free tier available)
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b

   # Keep embeddings local and free (requires Ollama running)
   EMBEDDING_PROVIDER=ollama
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

   You can also use `openai` or `anthropic`. See [Configuration](#️-configuration) below.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 🦙 Quick Start (Fully Local, 100% Offline)

Run everything on your own machine without paying for API calls.

1. **Install Ollama** from [ollama.com](https://ollama.com).

2. **Pull the required models**

   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

3. **Configure `.env`**

   ```env
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.1:8b
   EMBEDDING_PROVIDER=ollama
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📖 How to Use

1. **Upload a PDF**: Use the sidebar to upload a research paper (max 25 MB).
2. **Summary tab**: Type `summarize the paper`. The app indexes the document and generates a section-by-section summary with page citations.
3. **Concepts tab**: Type `what are the contributions?` to extract methods, metrics, and datasets.
4. **Slides tab**: Type `make slides`. Wait a few seconds, then click **Download PPTX** to get a deck with speaker notes.
5. **Viva tab**: Type `quiz me`. Answer the questions in the text box. The agent grades your answers, asks follow-ups, and generates a final study report (Markdown download).
6. **Ask tab**: Ask any specific question, e.g. `What dataset did they use?`. If the paper doesn't mention it, the agent replies "The paper does not state this."

---

## ⚙️ Configuration

All configuration is handled through the `.env` file.

| Variable | Options / Default | Description |
|----------|-------------------|-------------|
| `LLM_PROVIDER` | `groq`, `openai`, `anthropic`, `ollama` | The LLM engine to use. |
| `GROQ_API_KEY` | – | Required when `LLM_PROVIDER=groq`. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name. Check the Groq Console for current models. |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | – / `gpt-4o-mini` | Required when `LLM_PROVIDER=openai`. |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | – | Both required when `LLM_PROVIDER=anthropic`. Use a current model name from Anthropic's docs. |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.1:8b` | Local model settings. |
| `EMBEDDING_PROVIDER` | `ollama`, `openai` | Used to vectorize the PDF. Local Ollama is recommended to save cost. |
| `CHUNK_SIZE` | `1500` | Size of text chunks stored in ChromaDB. |
| `MAX_SLIDES` | `15` | Maximum number of slides to generate. |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF size. |

---

## 🛠️ Troubleshooting

- **Groq error: HTTP 404** — Groq deprecates models often. Check the Groq Console for the latest supported model and update `GROQ_MODEL` in `.env`.
- **Tool call validation failed (Groq)** — Make sure you are on the latest version of this repo. Structured output for Groq uses `json_schema` mode to avoid this error.
- **Ollama not running** — Open the Ollama desktop app, or run `ollama serve` in a terminal.
- **`scanned PDF not supported`** — The parser needs a text-selectable PDF. Scanned images with no selectable text are rejected.
- **Responses are very slow** — Local 7B/8B models can take 30–60 seconds per request on CPU. For faster responses, switch to Groq in `.env`.
- **Changed the embedding model and results look wrong** — Each document is indexed per embedding model, so re-upload the PDF after switching `EMBEDDING_PROVIDER` or the embedding model.

---

## 🧪 Testing

The project uses `pytest` with offline mocks (no network calls during tests).

```bash
pytest -v
```

To run the faithfulness evals on the golden paper (added in Phase 8):

```bash
python evals/run_evals.py
```

---

## 🐳 Docker Deployment

> Docker files are delivered in Phase 8.

```bash
docker compose up --build
```

To run with a local Ollama container:

```bash
docker compose --profile local up --build
```

When using the bundled Ollama service, set `OLLAMA_BASE_URL=http://ollama:11434` in `.env`.

---

## 🏗️ Architecture & Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain Expression Language (LCEL), `RunnableBranch`
- **Vector DB**: ChromaDB (`PersistentClient`)
- **PDF Parsing**: PyMuPDF / `pymupdf4llm`
- **Slides**: `python-pptx`
- **Validation**: Pydantic v2 (strict schemas on all LLM outputs)
- **Testing**: `pytest`, `pytest-mock`

For the full design, module contracts and build phases, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 🔒 Notes

- Text from uploaded papers is treated as untrusted data. It is wrapped in delimiters and never followed as instructions.
- API keys live only in `.env`, which is git-ignored. Never commit it.
