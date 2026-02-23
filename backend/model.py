# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

"""
Model interface for the chat backend.
This module provides a unified interface for different model providers.
"""
from typing import List, Dict
from backend.model_factory import model_provider
import logging

logger = logging.getLogger(__name__)

from backend.config import config

def generate(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Generate a response for the given list of messages.
    
    This function abstracts away the underlying model provider,
    allowing the frontend to remain agnostic about whether we're
    using a local model or an external API.
    
    Args:
        messages: A list of message dictionaries, e.g., [{"role": "user", "content": "Hello"}]
        
    Returns:
        Generated response dictionary with 'response' and 'finish_reason' keys
    """
    try:
        logger.info(f"Generating response for {len(messages)} messages.")
        
        # Route to agentic workflow if configured for OpenAI (or MedGemma when supported)
        if config.MODEL_PROVIDER in ["openai", "medgemma"]:
            logger.info("Routing request through LangGraph agent workflow.")
            from backend.agent import generate_agentic_response
            result = generate_agentic_response(messages)
        else:
            logger.info("Routing request through standard provider workflow.")
            result = model_provider.generate(messages)
            
        logger.info(f"Generated response length: {len(result.get('response', ''))}")
        return result
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
