"""Local model provider using Transformers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base import ModelProvider
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class LocalModelProvider(ModelProvider):
    """Local model provider using HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "microsoft/phi-2", max_tokens: int = 64):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local model and tokenizer."""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add pad token if it doesn't exist
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Determine device and dtype
            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = "auto"
            elif torch.backends.mps.is_available():
                dtype = torch.float32  # MPS does not support float16 well
                device_map = "auto"
            else:
                dtype = torch.float32
                device_map = "auto"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=dtype, 
                device_map=device_map
            )
            self._generator = pipeline(
                "text-generation", 
                model=self._model, 
                tokenizer=self._tokenizer
            )
            logger.info("Local model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert message history to a formatted prompt string."""
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            elif role == "system":
                formatted_parts.append(f"System: {content}")
        
        # Add the prompt for the assistant to respond
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)

    def generate(self,  messages: List[Dict[str, str]]) -> str:
        """Generate a response using the local model."""
        if not self.is_available():
            return "[Local model not available]"
        
        try:
            formatted_prompt = self._format_messages(messages)
            output = self._generator(
                formatted_prompt, 
                max_new_tokens=self.max_tokens, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id
            )
            
            if output and len(output) > 0:
                generated_text = output[0]["generated_text"]
                # Extract only the new response (after the last "Assistant:")
                if "Assistant:" in generated_text:
                    response_parts = generated_text.split("Assistant:")
                    if len(response_parts) > 1:
                        response = response_parts[-1].strip()
                        # Clean up any extra formatting
                        response = response.split("\n\nHuman:")[0].strip()
                        return response if response else "[No response generated]"
                
                return "[No response generated]"
            else:
                return "[No response generated]"
                
        except Exception as e:
            logger.error(f"Error generating response with local model: {e}")
            return f"[Model error: {e}]"
    
    def is_available(self) -> bool:
        """Check if the local model is available."""
        return all([self._model, self._tokenizer, self._generator])
