import json
import re
from typing import Dict, List, Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import accelerate
import modal


app = modal.App(name="script_test2_app")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "accelerate"
)

@app.cls(
    image = image,
    gpu="A10G"  
)
class VideoScriptGenerator:
    """
    Video script generator using Hugging Face transformers with:
    - Structured JSON output
    - Multi-stage generation
    - Feedback-based refinement
    - Live script generation
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # Load tokenizer with padding settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            padding_side='left',  # Ensure padding is on the left side
            truncation_side='left'
        )
    
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        # Load model with additional configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            # Add pad token configuration
            pad_token_id=self.tokenizer.pad_token_id
        )

        
        self.system_prompt = """You are a professional video script generator. 
        Generate JSON output strictly following this structure:
        {
            "topic": "Topic Name",
            "description": "description of the video",
            "audio_script": [{
                "timestamp": "00:00",
                "text": "Narration text",
                "speaker": "default|narrator_male|narrator_female",
                "speed": 0.9-1.1,
                "pitch": 0.9-1.2,
                "emotion": "neutral|serious|dramatic|mysterious|informative"
            }],
            "visual_script": [{
                "timestamp_start": "00:00",
                "timestamp_end": "00:05",
                "prompt": "Detailed Stable Diffusion prompt",
                "negative_prompt": "Low quality elements to avoid",
                "style": "realistic|cinematic|hyperrealistic|fantasy|scientific",
                "guidance_scale": 11.0-14.0,
                "steps": 50-100,
                "seed": 6 digit integer,
                "width": 1024,
                "height": 576
            }]
        }
         ex1_json = {
  "topic": "How to Drive a Car",
  "description": "A step-by-step guide on driving a car safely and confidently.",
  "audio_script": [
      {
      "timestamp": "00:00",
      "text": "Driving a car is an essential skill that requires focus, patience, and practice.",
      "speaker": "narrator_male",
      "speed": 1.0,
      "pitch": 1.0,
      "emotion": "neutral"
      },
      {
      "timestamp": "00:05",
      "text": "Before starting the car, adjust your seat, mirrors, and ensure your seatbelt is fastened.",
      "speaker": "narrator_female",
      "speed": 1.0,
      "pitch": 1.1,
      "emotion": "informative"
      },
      {
      "timestamp": "00:15",
      "text": "Turn the ignition key or press the start button while keeping your foot on the brake.",
      "speaker": "narrator_male",
      "speed": 0.95,
      "pitch": 1.0,
      "emotion": "calm"
      },
      {
      "timestamp": "00:20",
      "text": "Slowly release the brake and gently press the accelerator to move forward.",
      "speaker": "narrator_female",
      "speed": 1.1,
      "pitch": 1.0,
      "emotion": "guiding"
      },
      {
      "timestamp": "00:25",
      "text": "Use the steering wheel to navigate while maintaining a steady speed.",
      "speaker": "narrator_male",
      "speed": 1.0,
      "pitch": 1.0,
      "emotion": "calm"
      }
  ],
  "visual_script": [
      {
      "timestamp_start": "00:00",
      "timestamp_end": "00:05",
      "prompt": "A person sitting in the driver's seat of a modern car, gripping the steering wheel and looking ahead. The dashboard is visible with standard controls.",
      "negative_prompt": "blurry, unrealistic interior, poor lighting",
      "style": "realistic",
      "guidance_scale": 11.5,
      "steps": 50,
      "seed": 123456,
      "width": 1024,
      "height": 576,
      "strength": 0.75
      },
      {
      "timestamp_start": "00:05",
      "timestamp_end": "00:15",
      "prompt": "A close-up of a driver's hands adjusting the side mirrors and fastening the seatbelt inside a well-lit car interior.",
      "negative_prompt": "cluttered background, distorted perspective",
      "style": "cinematic",
      "guidance_scale": 12.0,
      "steps": 60,
      "seed": 654321,
      "width": 1024,
      "height": 576,
      "strength": 0.8
      },
      {
      "timestamp_start": "15:00",
      "timestamp_end": "00:20",
      "prompt": "A driver's hand turning the ignition key or pressing the start button in a modern car with a digital dashboard.",
      "negative_prompt": "low detail, unrealistic lighting, old car model",
      "style": "hyperrealistic",
      "guidance_scale": 12.5,
      "steps": 70,
      "seed": 789101,
      "width": 1024,
      "height": 576,
      "strength": 0.85
      },
      {
      "timestamp_start": "00:20",
      "timestamp_end": "00:25",
      "prompt": "A slow-motion shot of a car's foot pedals as the driver releases the brake and presses the accelerator.",
      "negative_prompt": "blurry, cartoonish, extreme close-up",
      "style": "cinematic",
      "guidance_scale": 11.5,
      "steps": 75,
      "seed": 222333,
      "width": 1024,
      "height": 576,
      "strength": 0.8
      },
      {
      "timestamp_start": "00:25",
      "timestamp_end": "00:30",
      "prompt": "A wide-angle shot of a car moving smoothly on a suburban road, the driver confidently steering the wheel.",
      "negative_prompt": "chaotic traffic, bad weather, motion blur",
      "style": "realistic",
      "guidance_scale": 13.0,
      "steps": 50,
      "seed": 987654,
      "width": 1024,
      "height": 576,
      "strength": 0.75
      }
  ]
}
        Ensure audio and visual timestamps are synchronized.
        
        """

    def _generate_content(self, prompt: str) -> Generator[str, None, None]:
        # Prepare input
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.device)
        
        # Create a list to store generated chunks
        generated_chunks = []
        
        # Set up streaming
        def generate_stream():
            output = self.model.generate(
                input_ids=input_ids.input_ids,
                attention_mask=input_ids.attention_mask,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            )
            return output

        # Run generation in a thread
        generation_thread = Thread(target=generate_stream)
        generation_thread.start()
        
        # Yield chunks as they are generated
        for chunk in TextIteratorStreamer(self.tokenizer, skip_prompt=True):
            yield chunk
            generated_chunks.append(chunk)
        
        # Wait for thread to complete
        generation_thread.join()
        
        # Return full generated text
        return ''.join(generated_chunks)
    @modal.method()   
    def _extract_json(self, raw_text: str) -> Dict:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                return json.loads(json_match.group()) if json_match else {}
            except Exception as e:
                raise ValueError(f"JSON extraction failed: {str(e)}")
    @modal.method()
    def generate_script(self, topic: str, duration: int = 60, key_points: Optional[List[str]] = None) -> Generator[str, None, None]:
        prompt = f"""Generate a {duration}-second video script about: {topic}
        Key Points: {key_points or 'Comprehensive coverage'}
        - At least {duration//5} segments (5-second intervals)
        - Engaging and scientifically accurate narration
        - Cinematic visuals with detailed prompts"""
        
        buffer = ""
        for chunk in self._generate_content(prompt):
            buffer += chunk
            yield chunk  # Stream data as it's received
    @modal.method()
    def refine_script(self, existing_script: Dict, feedback: str) -> Generator[str, None, None]:
        prompt = f"""Refine this script based on feedback:
        Existing Script: {json.dumps(existing_script, indent=2)}
        Feedback: {feedback}
        Maintain structure, valid parameters, and timestamp continuity."""
        
        buffer = ""
        for chunk in self._generate_content(prompt):
            buffer += chunk
            yield chunk  # Stream refinement updates
    @modal.method()
    def save_script(self, script: Dict, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(script, f, indent=2)

@app.local_entrypoint()
def main():
    generator = VideoScriptGenerator()
    try:
        print("Generating Script...")
        script_chunks = generator.generate_script.remote_gen(
            topic="Hot Wheels: The Ultimate Collector’s Guide",
            duration=60,
            key_points=["History of Hot Wheels", "Rare models", "Future designs"]
        )

        full_script = ""
        for chunk in script_chunks:
            print(chunk, end="", flush=True)  # Print streaming data in real-time
            full_script += chunk

        script_json = generator._extract_json_remote_gen(full_script)
        generator.save_script_remote_gen(script_json, "scripts.json")

        feedback = input("\nProvide feedback (or type 'no' to skip refinement): ")
        if feedback.lower() != "no":
            print("Refining Script...")
            refined_chunks = generator.refine_script_remote_gen(script_json, feedback)

            full_refined_script = ""
            for chunk in refined_chunks:
                print(chunk, end="", flush=True)
                full_refined_script += chunk

            refined_json = generator._extract_json_remote_gen(full_refined_script)
            generator.save_script_remote_gen(refined_json, "scripts.json")
    except Exception as e:
        print(f"Script generation failed: {str(e)}")
        