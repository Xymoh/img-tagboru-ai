from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Optional
import urllib.error
import urllib.request

try:
    import ollama
except ImportError:
    ollama = None

# Import the tag vocabulary from the trained tagger
try:
    from backend.tagger import _download_model_file, _load_tags
    HAS_TAGGER = True
except ImportError:
    HAS_TAGGER = False


@dataclass(frozen=True)
class DescriptionTagResult:
    tags: list[str]
    raw_response: str
    model: str


class DescriptionTagger:
    """Generate Danbooru tags from text descriptions using local LLM (Ollama)."""
    
    DEFAULT_MODEL = "qwen2:7b"
    DEFAULT_HOST = "http://localhost:11434"
    MAX_TAGS = 40
    
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.danbooru_tags = self._load_danbooru_whitelist()
        self.known_tags = self._load_known_tags()
        self.system_prompt = self._build_system_prompt()
    
    def _load_known_tags(self) -> list[str]:
        """Load tag vocabulary from the trained image tagger as a set for fast lookup."""
        if not HAS_TAGGER:
            return set()
        
        try:
            from pathlib import Path
            tags_path = _download_model_file("selected_tags.csv")
            tag_records = _load_tags(tags_path)
            tags = set(record.name for record in tag_records)
            print(f"Loaded {len(tags)} known tags from trained tagger")
            return tags
        except Exception as e:
            print(f"Warning: Could not load trained tagger tags: {e}")
            return set()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with real tags from the trained model."""
        if not self.known_tags:
            # Fallback if no trained tags available
            return self._get_default_system_prompt()
        
        # Create tag sets by category with more examples
        character_tags = sorted([t for t in self.known_tags if any(x in t for x in ['girl', 'boy', '1girl', '2girls', '1boy', 'boy', 'hair', 'eyes'])])[:20]
        clothing_tags = sorted([t for t in self.known_tags if any(x in t for x in ['dress', 'shirt', 'skirt', 'coat', 'outfit', 'top', 'bottom', 'uniform'])])[:15]
        setting_tags = sorted([t for t in self.known_tags if any(x in t for x in ['indoors', 'outdoors', 'beach', 'forest', 'room', 'sky', 'bedroom', 'office'])])[:12]
        action_tags = sorted([t for t in self.known_tags if any(x in t for x in ['standing', 'sitting', 'lying', 'dancing', 'running', 'holding', 'reading', 'playing'])])[:12]
        style_tags = sorted([t for t in self.known_tags if any(x in t for x in ['painting', 'detailed', 'quality', 'lighting', 'style', 'realistic', 'sketch', 'watercolor'])])[:12]
        
        prompt = f"""You are an expert AI art prompt generator trained on real image tagging data from a neural network tagger.
Generate tags from description using this real trained vocabulary:
You are an image tagging AI. Output 30+ tags for the description.
Generate tags. Use these trained vocabulary examples:
Generate 40-50 detailed image tags from description. Use real trained vocabulary.

EXAMPLES OF TRAINED TAGS YOU CAN USE:
{', '.join(character_tags[:12])} {', '.join(clothing_tags[:12])} {', '.join(setting_tags[:10])} {', '.join(action_tags[:10])}

INSTRUCTION:
- Generate 40-50 comma-separated tags (not 5, not 10 - aim for 40+)
- Use tags similar to examples above
- Format: tag1, tag2, tag3, ...
- Output ONLY tags, no explanations

EXAMPLE OUTPUT (THIS FORMAT):
1girl, long_hair, black_hair, blue_eyes, nun, habit, small_breasts, standing, indoors, church, detailed, beautiful, realistic, high_quality, masterpiece, and many more"""
        
        return prompt
    
    def _get_default_system_prompt(self) -> str:
        """Fallback system prompt when trained tags aren't available."""
        return """You are an expert AI art prompt generator.

Convert the user's description into a detailed comma-separated prompt for image generation.

STYLE:
- Be specific: character details, clothing, pose, environment
- Include: lighting, atmosphere, artistic style, quality level
- Use underscores for multi-word tags

OUTPUT:
- Single comma-separated list, 15-40 tags
- Lowercase, no explanations or code blocks
- Order by importance

Example input: "girl sitting by a window"
Example output: 1girl, sitting, window, books, warm_lighting, cozy, detailed, soft_colors"""
    
    
    def _load_danbooru_whitelist(self) -> set[str]:
        """Load valid Danbooru tag names from CSV file into a set for O(1) lookup.
        
        Returns:
            Set of valid Danbooru tag names (lowercase)
            
        Returns empty set if CSV file not found (graceful fallback).
        """
        csv_path = os.path.join(os.path.dirname(__file__), "..", "danbooru_tags_post_count.csv")
        
        if not os.path.exists(csv_path):
            # Fallback: CSV not found, return empty set (all tags will be accepted)
            return set()
        
        tags = set()
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_name = row.get("name", "").strip().lower()
                    if tag_name:
                        tags.add(tag_name)
            print(f"Loaded {len(tags)} Danbooru tags from whitelist")
            return tags
        except Exception as e:
            print(f"Warning: Could not load Danbooru whitelist: {e}")
            return set()
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3) as response:
                response.read()
            return True
        except Exception:
            return False
    
    def list_available_models(self) -> list[str]:
        """Get list of available local models."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))

            models = []
            for item in payload.get("models", []):
                model_name = item.get("model") or item.get("name")
                if model_name:
                    models.append(model_name)
            return models
        except Exception:
            try:
                response = self.client.list()
                return [m.model for m in response.models if getattr(m, "model", None)]
            except Exception:
                return []
    
    def pull_model(self, model: str) -> None:
        """Download a model from Ollama registry."""
        try:
            self.client.pull(model)
        except Exception as e:
            raise RuntimeError(f"Failed to pull model {model}: {e}")
    
    def generate_tags(self, description: str) -> DescriptionTagResult:
        """Generate Danbooru tags from a text description.
        
        Args:
            description: Text description of what you want to see
            
        Returns:
            DescriptionTagResult with generated tags and raw response
            
        Raises:
            RuntimeError: If Ollama is not running or model generation fails
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=description,
                system=self.system_prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "num_predict": 150,
                },
                stream=False,
            )
            
            raw_text = response.get("response", "").strip()
            tags = self._parse_tags(raw_text)
            
            return DescriptionTagResult(
                tags=tags,
                raw_response=raw_text,
                model=self.model,
            )
        except Exception as e:
            raise RuntimeError(f"Tag generation failed: {e}")
    
    def _parse_tags(self, raw_response: str) -> list[str]:
        """Extract tags from raw LLM response, filtered to trained vocabulary.
        
        Constrains output to only tags from the trained image tagger's vocabulary,
        ensuring compatibility with the actual model's understanding.
        """
        raw_response = self._clean_response_text(raw_response)
        tags = []
        seen = set()
        for chunk in re.split(r"[\n,\.]+", raw_response):
            line = chunk.strip()
            if not line:
                continue
            # Remove numbered list markers like "1. ", "2) " but preserve tag numbers like "1girl"
            line = re.sub(r"^[0-9]+[.)]\s*|^[\-\*]\s+", "", line)
            line = line.strip().strip(".")
            if not line or any(c in line for c in [":", "=", ">"]):
                continue
            normalized = line.lower()
            if normalized in seen:
                continue
            
            # Filter to only known tags from trained vocabulary
            if self.known_tags and normalized not in self.known_tags:
                continue
            
            seen.add(normalized)
            tags.append(normalized)
            if len(tags) >= DescriptionTagger.MAX_TAGS:
                break
        return tags

    @staticmethod
    def _clean_response_text(raw_response: str) -> str:
        """Remove common wrapper formats from model output before parsing."""
        text = raw_response.strip()

        # Prefer the contents of the first \boxed{...} block if present.
        text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text, flags=re.DOTALL)

        # Remove common markdown/code fences and latex-like wrappers.
        text = re.sub(r"```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        text = re.sub(r"\\(?:text|mathrm|mathbf|mathtt|operatorname)\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
        text = text.replace("{", " ").replace("}", " ")

        return text.strip()


def get_description_tagger(
    host: str = DescriptionTagger.DEFAULT_HOST,
    model: str = DescriptionTagger.DEFAULT_MODEL,
) -> DescriptionTagger:
    """Factory function to get a description tagger instance."""
    return DescriptionTagger(host=host, model=model)
