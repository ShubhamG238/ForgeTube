import json
import modal
from io import BytesIO

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "accelerate"
)

app = modal.App(name="script_test_app")

@app.function(image=image, gpu="A10G")
def script_generator(prompt: str) -> str:
    
    messages = [
        {"role": "youtube content creator", "content": prompt}
    ]

    # Use a pipeline as a high-level helper
    from transformers import pipeline
    pipe = pipeline("text-generation", model="google/gemma-2b")
    # Pass the messages (or just the prompt depending on your model's requirements)
    result = pipe(messages)
    return result[0]["generated_text"] 