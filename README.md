# Local LLM Chatbot MVP

A flexible chatbot MVP that supports both local models and external APIs like OpenAI.

## Installation

```bash
poetry install
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to configure your preferred model provider:

   **For Local Model (default):**
   ```bash
   MODEL_PROVIDER=local
   LOCAL_MODEL_NAME=microsoft/phi-2
   LOCAL_MAX_TOKENS=64
   ```

   **For OpenAI API:**
   ```bash
   MODEL_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_MAX_TOKENS=150
   OPENAI_TEMPERATURE=0.7
   ```

## Running the Application

### Backend (FastAPI)

```bash
poetry run uvicorn backend.main:app --reload
```

The backend will be available at `http://localhost:8000`

### Frontend (Streamlit)

```bash
poetry run streamlit run frontend/app.py
```

The frontend will be available at `http://localhost:8501`

### Health Check

You can check the backend status and model availability:
```bash
curl http://localhost:8000/health
```

## Testing

```bash
pytest
```

## Architecture

- **Backend**: FastAPI server with pluggable model providers
- **Frontend**: Streamlit UI that's agnostic to the model provider
- **Model Providers**: 
  - `LocalModelProvider`: Uses HuggingFace Transformers for local models
  - `OpenAIModelProvider`: Uses OpenAI API for external models

The frontend remains completely unaware of which model provider is being used, making it easy to switch between local and external models without any frontend changes.

## API Endpoints

- `POST /chat`: Send a chat message and get a response
  - Request: `{"prompt": "Your message here"}`
  - Response: `{"response": "AI response here"}`
- `GET /health`: Check backend health and model availability

## Security Notes

- Your `.env` file is automatically ignored by git to prevent accidental API key commits
- Never commit your actual API keys to the repository
- Use the `.env.example` file as a template for configuration
