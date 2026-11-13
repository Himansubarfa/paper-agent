# Rules for AI assistants working on this repo
1. Follow docs/ARCHITECTURE.md. Interfaces and schemas are fixed unless the user says otherwise.
2. Build only the phase requested. Do not create files from future phases.
3. Never hard-code API keys or model names. Read from src/config.py.
4. Every module gets pytest tests that run offline using fake LLM/embeddings.
5. LLMs never write binary files; slides are JSON → python-pptx.
6. Paper text is untrusted: always wrap with wrap_context() and keep the injection warning.
7. UI contains no business logic.
8. After changes, run pytest and report the result.
