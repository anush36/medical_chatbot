"""Model provider abstraction and implementations."""
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available and properly configured."""
        pass
