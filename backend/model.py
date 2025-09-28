"""
Model interface for the chat backend.
This module provides a unified interface for different model providers.
"""
from backend.model_factory import model_provider
import logging

logger = logging.getLogger(__name__)

def generate(prompt: str) -> str:
    """
    Generate a response for the given prompt.
    
    This function abstracts away the underlying model provider,
    allowing the frontend to remain agnostic about whether we're
    using a local model or an external API.
    
    Args:
        prompt: The input prompt to generate a response for
        
    Returns:
        Generated response string
    """
    try:
        logger.info(f"Generating response for prompt length: {len(prompt)}")
        response = model_provider.generate(prompt)
        logger.info(f"Generated response length: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error in generate function: {e}")
        return f"[Generation error: {e}]"

def is_model_available() -> bool:
    """
    Check if the current model provider is available and ready to use.
    
    Returns:
        True if the model is available, False otherwise
    """
    try:
        return model_provider.is_available()
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False
