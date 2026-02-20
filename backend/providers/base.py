"""Model provider abstraction and implementations."""
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate(self, messages: list) -> dict:
        """Generate a response, returning a dict with 'response' and 'finish_reason' keys."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available and properly configured."""
        pass
