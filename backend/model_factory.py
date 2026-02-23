# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

"""Model factory for creating the appropriate model provider."""
from backend.config import config
from backend.providers import ModelProvider, LocalModelProvider, OpenAIModelProvider
import logging

logger = logging.getLogger(__name__)

def create_model_provider() -> ModelProvider:
    """Create and return the appropriate model provider based on configuration."""
    
    if config.MODEL_PROVIDER == "openai":
        logger.info("Creating OpenAI model provider")
        return OpenAIModelProvider(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            max_tokens=config.OPENAI_MAX_TOKENS,
            temperature=config.OPENAI_TEMPERATURE
        )
    elif config.MODEL_PROVIDER == "medgemma":
            logger.info("Creating MedGemma (vLLM) provider")
            return OpenAIModelProvider(
                api_key=config.MEDGEMMA_API_KEY,
                base_url=config.MEDGEMMA_BASE_URL,
                model=config.MEDGEMMA_MODEL,
                max_tokens=config.OPENAI_MAX_TOKENS,
                temperature=config.OPENAI_TEMPERATURE
            )
    elif config.MODEL_PROVIDER == "local":
        logger.info("Creating local model provider")
        return LocalModelProvider(
            model_name=config.LOCAL_MODEL_NAME,
            max_tokens=config.LOCAL_MAX_TOKENS
        )
    else:
        logger.error(f"Unknown model provider: {config.MODEL_PROVIDER}")
        raise ValueError(f"Unsupported model provider: {config.MODEL_PROVIDER}")

# Create the global model provider instance
model_provider = create_model_provider()
