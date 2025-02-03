import json
import re
from typing import Dict, List, Optional
from ollama import chat

class VideoScriptGenerator:
    """
    Video script generator using Ollama with:
    - Structured JSON output
    - Multi-stage generation
    - Feedback-based refinement
    - Streaming API integration
    """
    
    def __init__(self, model: str = 'gemmascript'):
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
        ex1_json = {
        "topic": "How to Drive a Car",
        "audio_script": [
            {
            "timestamp": "00:00:00",
            "text": "Driving a car is an essential skill that requires focus, patience, and practice.",
            "speaker": "narrator_male",
            "speed": 1.0,
            "pitch": 1.0,
            "emotion": "neutral"
            },
            {
            "timestamp": "00:05:00",
            "text": "Before starting the car, adjust your seat, mirrors, and ensure your seatbelt is fastened.",
            "speaker": "narrator_female",
            "speed": 1.0,
            "pitch": 1.1,
            "emotion": "informative"
            },
            {
            "timestamp": "00:10:00",
            "text": "Turn the ignition key or press the start button while keeping your foot on the brake.",
            "speaker": "narrator_male",
            "speed": 0.95,
            "pitch": 1.0,
            "emotion": "calm"
            },
            {
            "timestamp": "00:15:00",
            "text": "Slowly release the brake and gently press the accelerator to move forward.",
            "speaker": "narrator_female",
            "speed": 1.1,
            "pitch": 1.0,
            "emotion": "guiding"
            },
            {
            "timestamp": "00:20:00",
            "text": "Use the steering wheel to navigate while maintaining a steady speed.",
            "speaker": "narrator_male",
            "speed": 1.0,
            "pitch": 1.0,
            "emotion": "calm"
            }
        ],
        "visual_script": [
            {
            "timestamp": "00:00:00",
            "prompt": "A person sitting in the driver's seat of a modern car, gripping the steering wheel and looking ahead. The dashboard is visible with standard controls.",
            "negative_prompt": "blurry, unrealistic interior, poor lighting",
            "style": "realistic",
            "guidance_scale": 7.5,
            "steps": 50,
            "seed": 123456,
            "width": 1024,
            "height": 576,
            "strength": 0.75
            },
            {
            "timestamp": "00:05:00",
            "prompt": "A close-up of a driver's hands adjusting the side mirrors and fastening the seatbelt inside a well-lit car interior.",
            "negative_prompt": "cluttered background, distorted perspective",
            "style": "cinematic",
            "guidance_scale": 8.0,
            "steps": 60,
            "seed": 654321,
            "width": 1024,
            "height": 576,
            "strength": 0.8
            },
            {
            "timestamp": "00:10:00",
            "prompt": "A driver's hand turning the ignition key or pressing the start button in a modern car with a digital dashboard.",
            "negative_prompt": "low detail, unrealistic lighting, old car model",
            "style": "hyperrealistic",
            "guidance_scale": 9.0,
            "steps": 70,
            "seed": 789101,
            "width": 1024,
            "height": 576,
            "strength": 0.85
            },
            {
            "timestamp": "00:15:00",
            "prompt": "A slow-motion shot of a car's foot pedals as the driver releases the brake and presses the accelerator.",
            "negative_prompt": "blurry, cartoonish, extreme close-up",
            "style": "cinematic",
            "guidance_scale": 8.5,
            "steps": 75,
            "seed": 222333,
            "width": 1024,
            "height": 576,
            "strength": 0.8
            },
            {
            "timestamp": "00:20:00",
            "prompt": "A wide-angle shot of a car moving smoothly on a suburban road, the driver confidently steering the wheel.",
            "negative_prompt": "chaotic traffic, bad weather, motion blur",
            "style": "realistic",
            "guidance_scale": 7.0,
            "steps": 50,
            "seed": 987654,
            "width": 1024,
            "height": 576,
            "strength": 0.75
            }
        ]
    }

        """
    
    def _generate_content(self, prompt: str) -> str:
        stream = chat(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_prompt},
                      {'role': 'user', 'content': prompt}],
            stream=True
        )
        
        response_text = ""
        for chunk in stream:
            response_text += chunk['message']['content']
        
        return response_text
    
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
    
    def generate_script(self, topic: str, duration: int = 60, key_points: Optional[List[str]] = None) -> Dict:
        prompt = f"""Generate a {duration}-second video script about: {topic}
        Key Points: {key_points or 'Comprehensive coverage'}
        - At least {duration//5} segments (5-second intervals)
        - Engaging and scientifically accurate narration
        - Cinematic visuals with detailed prompts"""
        
        raw_output = self._generate_content(prompt)
        return self._extract_json(raw_output)
    
    def refine_script(self, existing_script: Dict, feedback: str) -> Dict:
        prompt = f"""Refine this script based on feedback:
        Existing Script: {json.dumps(existing_script, indent=2)}
        Feedback: {feedback}
        Maintain structure, valid parameters, and timestamp continuity."""
        
        raw_output = self._generate_content(prompt)
        return self._extract_json(raw_output)
    
    def save_script(self, script: Dict, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(script, f, indent=2)

# Example Usage
if __name__ == "__main__":
    generator = VideoScriptGenerator()
    try:
        script = generator.generate_script(
            topic="Hot Wheels: The Ultimate Collector’s Guide",
            duration=90,
            key_points=["History of Hot Wheels", "Rare models", "Future designs"]
        )
        print("Generated Script:")
        print(json.dumps(script, indent=2))

        feedback = input("Provide feedback (or type 'no' to skip refinement): ")
        if feedback.lower() != "no":
            refined_script = generator.refine_script(script, feedback)
            print("\nRefined Script:")
            print(json.dumps(refined_script, indent=2))
            generator.save_script(refined_script, "refined_hotwheels_script.json")
        else:
            generator.save_script(script, "hotwheels_script.json")
    except Exception as e:
        print(f"Script generation failed: {str(e)}")