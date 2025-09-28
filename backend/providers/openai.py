"""OpenAI model provider."""
from openai import OpenAI
from .base import ModelProvider
import logging

logger = logging.getLogger(__name__)

class OpenAIModelProvider(ModelProvider):
    """OpenAI model provider using the OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 150, temperature: float = 0.7):
        self.api_key = api_key
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
            self._client = OpenAI(api_key=self.api_key)
            print(self.api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    
    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI API."""
        if not self.is_available():
            return "[OpenAI API not available - check your API key configuration]"
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
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
