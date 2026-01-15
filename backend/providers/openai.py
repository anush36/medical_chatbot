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
        # Allow initialization even if key is fake, as Cloud Run might rely on IAM instead
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url 
            )
            logger.info(f"OpenAI client initialized (URL: {self.base_url or 'Default OpenAI'})")
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")

    def _get_gcp_token(self) -> Optional[str]:
        """Fetch Google ID token if running against Cloud Run."""
        try:
            import google.auth
            import google.auth.transport.requests
            from google.oauth2 import id_token
            
            # Setup the request to fetch the token
            auth_req = google.auth.transport.requests.Request()
            # The 'audience' must match the service URL exactly
            target_audience = self.base_url.split("/v1")[0] if "/v1" in self.base_url else self.base_url
            
            token = id_token.fetch_id_token(auth_req, target_audience)
            return token
        except ImportError:
            logger.warning("google-auth library not found. Cannot auto-authenticate with Cloud Run.")
            return None
        except Exception as e:
            # This often happens if you aren't logged in via gcloud
            logger.debug(f"Could not fetch GCP token (this is normal for standard OpenAI usage): {e}")
            return None
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using OpenAI API."""
        if not self.is_available():
            return "[OpenAI API not available - check your API key configuration]"
        
        try:
            openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            # Prepare arguments
            request_kwargs = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # AUTO-AUTH: If this is a Cloud Run URL, try to inject a Google Identity Token
            if self.base_url and "run.app" in self.base_url:
                token = self._get_gcp_token()
                if token:
                    # Inject the token into the headers, overriding the 'fake-key'
                    request_kwargs["extra_headers"] = {
                        "Authorization": f"Bearer {token}"
                    }
            
            response = self._client.chat.completions.create(**request_kwargs)
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "[No response generated]"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"[Model Error: {e}]"
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._client is not None