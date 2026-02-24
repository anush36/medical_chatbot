# Copyright (c) 2026 Anush Agarwal. All rights reserved.
# This code is proprietary and provided for public review and educational purposes.
# Unauthorized use, reproduction, or distribution is strictly prohibited.

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
        self._gcp_token_cache = {"token": None, "expires_at": 0}
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
        """Fetch Google ID token if running against Cloud Run. Caches token to avoid repeated slow fetches."""
        import time
        current_time = time.time()
        
        # Check cache first
        if "token" in self._gcp_token_cache and self._gcp_token_cache["expires_at"] > current_time:
            return self._gcp_token_cache["token"]
            
        try:
            import google.auth
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token
            
            if self.base_url:
                target_audience = self.base_url.split("/v1")[0] if "/v1" in self.base_url else self.base_url
            else:
                target_audience = ""
                
            try:
                # 1. Try standard GCP method (works when deployed on Cloud Run / Compute Engine)
                req = Request()
                token = id_token.fetch_id_token(req, target_audience)
                if token:
                    self._gcp_token_cache["token"] = token
                    self._gcp_token_cache["expires_at"] = current_time + 3000 # Cache for 50 mins
                    return token
            except Exception as e:
                logger.debug(f"Direct ID token fetch failed (expected if running locally): {e}")

            # 2. Fallback for Local Development (uses gcloud CLI)
            import subprocess
            logger.info("Falling back to gcloud CLI for Identity Token...")
            result = subprocess.run(
                ["gcloud", "auth", "print-identity-token"],
                capture_output=True,
                text=True,
                check=True
            )
            token = result.stdout.strip()
            self._gcp_token_cache["token"] = token
            self._gcp_token_cache["expires_at"] = current_time + 3000 # Cache for 50 mins
            return token
            
        except ImportError:
            logger.warning("google-auth library not found. Cannot auto-authenticate with Cloud Run.")
            return None
        except Exception as e:
            logger.error(f"Could not fetch GCP ID token (Auth will fail if endpoint expects it): {e}")
            return None
    
    def generate(self, messages: List[Dict[str, any]]) -> Dict[str, str]:
        """Generate a response using OpenAI API, supporting both text and multimodal content."""
        if not self.is_available():
            return {"response": "[OpenAI API not available - check your API key configuration]", "finish_reason": "error"}
        
        try:
            # Prepare messages, ensuring we don't assume 'content' is always a string
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
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
                choice = response.choices[0]
                return {
                    "response": choice.message.content.strip(),
                    "finish_reason": choice.finish_reason or "unknown"
                }
            else:
                return {"response": "[No response generated]", "finish_reason": "error"}
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"response": f"[Model Error: {e}]", "finish_reason": "error"}
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._client is not None