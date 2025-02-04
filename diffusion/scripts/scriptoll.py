import json
import re
from typing import Dict, List, Optional, Generator
from ollama import chat

class VideoScriptGenerator:
    """
    Video script generator using Ollama with:
    - Structured JSON output
    - Multi-stage generation
    - Feedback-based refinement
    
    - Live script generation
    """
    
    def __init__(self, model: str = 'llama3.1'):
        self.model = model
        self.system_prompt = """You are a professional video script generator. 
        Generate JSON output strictly following this structure:
        {
            "topic": "Topic Name",
            "audio_script": [{
                "timestamp": "00:00",
                "text": "Narration text",
                "speaker": "default|narrator_male|narrator_female",
                "speed": 0.9-1.1,
                "pitch": 0.9-1.2,
                "emotion": "neutral|serious|dramatic|mysterious|informative"
            }],
            "visual_script": [{
                "timestamp": "00:00",
                "prompt": "Detailed Stable Diffusion prompt",
                "negative_prompt": "Low quality elements to avoid",
                "style": "realistic|cinematic|hyperrealistic|fantasy|scientific",
                "guidance_scale": 7.0-12.0,
                "steps": 50-100,
                "seed": 6-7 digit integer,
                "width": 1024,
                "height": 576
            }]
        }
        Ensure audio and visual timestamps are synchronized.
        """
    
    def _generate_content(self, prompt: str) -> Generator[str, None, None]:
        stream = chat(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_prompt},
                      {'role': 'user', 'content': prompt}],
            stream=True
        )
        
        for chunk in stream:
            yield chunk['message']['content']
    
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
    
    def refine_script(self, existing_script: Dict, feedback: str) -> Generator[str, None, None]:
        prompt = f"""Refine this script based on feedback:
        Existing Script: {json.dumps(existing_script, indent=2)}
        Feedback: {feedback}
        Maintain structure, valid parameters, and timestamp continuity."""
        
        buffer = ""
        for chunk in self._generate_content(prompt):
            buffer += chunk
            yield chunk  # Stream refinement updates
    
    def save_script(self, script: Dict, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(script, f, indent=2)

# Example Usage
if __name__ == "__main__":
    generator = VideoScriptGenerator()
    try:
        print("Generating Script...")
        script_chunks = generator.generate_script(
            topic="Hot Wheels: The Ultimate Collector’s Guide",
            duration=1,
            key_points=["History of Hot Wheels", "Rare models", "Future designs"]
        )
        
        full_script = ""
        for chunk in script_chunks:
            print(chunk, end="", flush=True)  # Print streaming data in real time
            full_script += chunk
        
        script_json = generator._extract_json(full_script)
        generator.save_script(script_json, "hotwheels_script.json")

        feedback = input("\nProvide feedback (or type 'no' to skip refinement): ")
        if feedback.lower() != "no":
            print("Refining Script...")
            refined_chunks = generator.refine_script(script_json, feedback)
            
            full_refined_script = ""
            for chunk in refined_chunks:
                print(chunk, end="", flush=True)
                full_refined_script += chunk
            
            refined_json = generator._extract_json(full_refined_script)
            generator.save_script(refined_json, "refined_hotwheels_script.json")
    except Exception as e:
        print(f"Script generation failed: {str(e)}")
