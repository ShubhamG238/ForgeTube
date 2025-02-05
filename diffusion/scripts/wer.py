import json
import re
import google.generativeai as genai
from typing import Dict, List, Optional

class VideoScriptGenerator:
    """
    A robust video script generation system using Gemini API with:
    - Structured JSON output with validation
    - Multi-stage generation process
    - Feedback-based refinement
    - Comprehensive error handling
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-exp-1206')
        
        self.system_prompt = """You are a professional video script generator for educational and marketing content.
        Generate JSON output with these strict rules:
        
        1. Structure:
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
                "guidance_scale": 11.0-14.0,
                "steps": 50-100,
                "seed": 6 digit integer,
                "width": 1024,
                "height": 576
            }]
        }
        
        2. Requirements:
        - Maintain timestamp synchronization between audio/visual
        - Ensure visual continuity through seed values
        - Vary speakers and emotions appropriately
        - Use technical parameters within specified ranges
        - Include detailed negative prompts
        - Validate JSON structure before output"""
    
    # Handling API calls
    def _generate_content(self, prompt: str, sys: str = None) -> str:
        try:
            contents = [prompt] if sys is None else [sys, prompt]
            response = self.model.generate_content(contents=contents)
            return response.text
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")

    
    # JSON extraction with multiple fallback strategies
    def _extract_json(self, raw_text: str) -> Dict:
        try:
            # First try direct parsing
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                # Attempt to extract JSON from markdown
                json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                # Fallback to bracket matching
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                return json.loads(json_match.group()) if json_match else {}
            except Exception as e:
                raise ValueError(f"JSON extraction failed: {str(e)}")
    
    # Validation of script structure and values
    def _validate_script(self, script: Dict) -> bool:
        required = {
            'audio_script': ['timestamp', 'text', 'speaker', 'speed', 'pitch', 'emotion'],
            'visual_script': ['timestamp', 'prompt', 'negative_prompt', 'style',
                             'guidance_scale', 'steps', 'seed', 'width', 'height']
        }
        
        for section, fields in required.items():
            if section not in script:
                raise ValueError(f"Missing section: {section}")
            for item in script[section]:
                for field in fields:
                    if field not in item:
                        raise ValueError(f"Missing field '{field}' in {section}")
                    
                # Validate parameter ranges
                if section == 'audio_script':
                    if not 0.9 <= item['speed'] <= 1.1:
                        raise ValueError("Speed must be between 0.9-1.1")
                elif section == 'visual_script':
                    if not 11.0 <= item['guidance_scale'] <= 14.0:
                        raise ValueError("Guidance scale must be 11.0-14.0")
        
        return True
    
    # Generates a complete video script with synchronized audio/visual elements
    def generate_script(self, topic: str, duration: int = 60, 
                       key_points: Optional[List[str]] = None) -> Dict:
        prompt = f"""Generate a {duration}-second video script about: {topic}
        Key Points: {key_points or 'Comprehensive coverage'}
        Requirements:
        - At least {duration//5} segments (5-second intervals)
        - Scientific accuracy with engaging delivery
        - Cinematic visual descriptions
        - Detailed negative prompts
        - Seed continuity for visual elements"""
        
        raw_output = self._generate_content(prompt, self.system_prompt)
        script = self._extract_json(raw_output)
        self._validate_script(script)
        return script
    
    # Iteratively improve an existing script based on user feedback
    def refine_script(self, existing_script: Dict, feedback: str) -> Dict:
        prompt = f"""Refine this script based on feedback:
        Existing Script: {json.dumps(existing_script, indent=2)}
        Feedback: {feedback}
        Requirements:
        - Maintain JSON structure
        - Preserve valid parameters
        - Ensure timestamp continuity"""
        
        raw_output = self._generate_content(prompt)
        refined_script = self._extract_json(raw_output)
        self._validate_script(refined_script)
        return refined_script
    

    def video_segmentation(self, existing_script: Dict):
        # Load the JSON file
        with open('script.json', 'r') as file:
            script_data = json.load(file)

        # Extract the visual_script part
        visual_script = script_data['visual_script']
        prompt=f"""Break down the following json into 2-3 key detailed visual descriptions. 
        Ensure each sub-prompt enhances specific elements of the original scene for better interpolation in video generation.
        visual prompt: '{visual_script}'
        Requirements:
        - devide each prompt into 2-3 prompts
        - Maintain JSON structure
        - Preserve valid parameters
        - Ensure timestamp continuity
        - Scientific accuracy with engaging delivery
        - Cinematic visual descriptions
        - Detailed negative prompts
        - Seed continuity for visual elements"""
        raw_output = self._generate_content(prompt)
        refined_script = self._extract_json(raw_output)
        return refined_script
        

    def save_script(self, script: Dict, filename: str) -> None:
        """Save validated script to file with indentation"""
        with open(filename, 'w') as f:
            json.dump(script, f, indent=2)

# Example Usage
if __name__ == "__main__":
    generator = VideoScriptGenerator(api_key="AIzaSyCidprNdzUho_rbZd4ZK9RR1TO0V3WxCqY")

    try:
        script = generator.generate_script(
            topic="Neural Networks in Medical Imaging",
            duration=90,
            key_points=["Diagnosis accuracy", "Pattern recognition", "Case studies"]
        )
        print("Initial Script:")
        print(json.dumps(script, indent=2))
        generator.save_script(script, "script.json")
        superref = generator.video_segmentation(script)
        print("\n segmented Script:")
        print(json.dumps(superref, indent=2))
        generator.save_script(superref, "super refined.json")
        feedback = input("Please provide feedback on the script (or type 'no' to skip refinement): ")

        if feedback.lower() != "no":
            refined_script = generator.refine_script(script, feedback)
            print("\nRefined Script:")
            print(json.dumps(refined_script, indent=2))
            generator.save_script(refined_script, "refined_medical_ai_script.json")

    except Exception as e:
        print(f"Script generation failed: {str(e)}")