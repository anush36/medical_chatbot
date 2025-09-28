"""Local model provider using Transformers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base import ModelProvider
import logging

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
    
    def generate(self, prompt: str) -> str:
        """Generate a response using the local model."""
        if not self.is_available():
            return "[Local model not available]"
        
        try:
            output = self._generator(
                prompt, 
                max_new_tokens=self.max_tokens, 
                do_sample=False, 
                temperature=1.0
            )
            return output[0]["generated_text"] if output else ""
        except Exception as e:
            logger.error(f"Error generating response with local model: {e}")
            return f"[Model error: {e}]"
    
    def is_available(self) -> bool:
        """Check if the local model is available."""
        return all([self._model, self._tokenizer, self._generator])
