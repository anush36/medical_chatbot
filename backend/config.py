# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

"""Configuration management for the chat backend."""
import os
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the chat backend."""
    
    # Model provider configuration
    MODEL_PROVIDER: Literal["local", "openai", "medgemma"] = os.getenv("MODEL_PROVIDER", "local")
    
    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "150"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # MedGemma (Cloud Run) Configuration
    # This is much clearer: "Where is my MedGemma running?"
    MEDGEMMA_BASE_URL: str = os.getenv("MEDGEMMA_BASE_URL", "")
    MEDGEMMA_API_KEY: str = os.getenv("MEDGEMMA_API_KEY", "fake-key") # vLLM often needs a non-empty key
    MEDGEMMA_MODEL: str = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")

    # Local model configuration
    LOCAL_MODEL_NAME: str = os.getenv("LOCAL_MODEL_NAME", "microsoft/phi-2")
    LOCAL_MAX_TOKENS: int = int(os.getenv("LOCAL_MAX_TOKENS", "64"))

# Global config instance
config = Config()
