"""
Model interface for the chat backend.
This module provides a unified interface for different model providers.
"""
from typing import List, Dict
from backend.model_factory import model_provider
import logging

logger = logging.getLogger(__name__)

def generate(messages: List[Dict[str, str]]) -> str:
    """
    Generate a response for the given list of messages.
    
    This function abstracts away the underlying model provider,
    allowing the frontend to remain agnostic about whether we're
    using a local model or an external API.
    
    Args:
        messages: A list of message dictionaries, e.g., [{"role": "user", "content": "Hello"}]
        
    Returns:
        Generated response string
    """
    try:
        logger.info(f"Generating response for {len(messages)} messages.")
        response = model_provider.generate(messages)
        logger.info(f"Generated response length: {len(response)}")
        return response
    except Exception as e:
        logger.error(f"Error during model generation: {e}", exc_info=True)
        # Re-raise the exception to be handled by the API endpoint layer,
        # which will return a proper HTTP 500 error.
        raise

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
