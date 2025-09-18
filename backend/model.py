from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_NAME = "microsoft/phi-2"

# Load model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
import platform
if torch.cuda.is_available():
    dtype = torch.float16
    device_map = "auto"
elif torch.backends.mps.is_available():
    dtype = torch.float32  # MPS does not support float16 well
    device_map = "auto"
else:
    dtype = torch.float32
    device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map=device_map)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate(prompt: str) -> str:
    # Use safe generation params for phi-2
    try:
        output = generator(prompt, max_new_tokens=64, do_sample=False, temperature=1.0)
        return output[0]["generated_text"] if output else ""
    except Exception as e:
        return f"[Model error: {e}]"
