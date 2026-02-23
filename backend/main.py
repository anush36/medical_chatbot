# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from backend.model import generate, is_model_available
from backend.config import config
from backend.pdf_parser import extract_text_from_pdf
import logging
from typing import List, Union, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MVP Chat Backend",
    description="A flexible chat backend supporting both local and external models",
    version="0.1.0"
)

from typing import List, Union, Dict, Any

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    finish_reason: str = "unknown"
    intermediate_steps: List[str] = []

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

@app.post("/parse-pdf")
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    """Endpoint to extract text from an uploaded PDF file."""
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error in parse-pdf endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Chat endpoint that generates responses using the configured model provider."""
    try:
        if not is_model_available():
            raise HTTPException(
                status_code=503, 
                detail=f"Model provider '{config.MODEL_PROVIDER}' is not available. Check your configuration."
            )
        messages = [msg.dict() for msg in request.messages]
        result = generate(messages)
        return ChatResponse(
            response=result.get("response", ""),
            finish_reason=result.get("finish_reason", "unknown"),
            intermediate_steps=result.get("intermediate_steps", [])
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
