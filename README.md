# Local LLM Chatbot MVP

## Installation

```bash
poetry install
```

## Running the Backend (FastAPI)

```bash
poetry run uvicorn backend.main:app --reload
```

## Running the Frontend (Streamlit)

```bash
poetry run streamlit run frontend/app.py
```

## Testing

```bash
pytest
```

---

- The backend exposes a `/chat` POST endpoint that takes `{prompt: str}` and returns `{response: str}`.
- The frontend provides a simple chat UI and calls the backend API.
- The backend loads a local LLM using Hugging Face transformers and accelerate.
