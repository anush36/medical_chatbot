"""OpenAI model provider."""
from openai import OpenAI
from .base import ModelProvider
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class OpenAIModelProvider(ModelProvider):
    """OpenAI-compatible model provider (works for OpenAI, vLLM, MedGemma, etc)."""    
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 150, temperature: float = 0.7):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            logger.warning("OpenAI API key not provided or using placeholder value")
            return
        
        try:
            # Pass the base_url to the client if it exists!
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url 
            )
            logger.info(f"OpenAI client initialized (URL: {self.base_url or 'Default OpenAI'})")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")

    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using OpenAI API."""
        if not self.is_available():
            return "[OpenAI API not available - check your API key configuration]"
        
        try:
            openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            response = self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "[No response generated]"
                
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"[OpenAI API error: {e}]"
    
    def is_available(self) -> bool:
        """Check if the OpenAI provider is available."""
        return (self._client is not None and 
                self.api_key and 
                self.api_key != "your_openai_api_key_here")
