from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.model import generate, is_model_available
from backend.config import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MVP Chat Backend",
    description="A flexible chat backend supporting both local and external models",
    version="0.1.0"
)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

class HealthResponse(BaseModel):
    status: str
    model_provider: str
    model_available: bool

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify the backend is running and model is available."""
    return HealthResponse(
        status="healthy",
        model_provider=config.MODEL_PROVIDER,
        model_available=is_model_available()
    )

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Chat endpoint that generates responses using the configured model provider."""
    try:
        if not is_model_available():
            raise HTTPException(
                status_code=503, 
                detail=f"Model provider '{config.MODEL_PROVIDER}' is not available. Check your configuration."
            )
        
        result = generate(request.prompt)
        return ChatResponse(response=result)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
