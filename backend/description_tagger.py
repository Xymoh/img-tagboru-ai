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
    
    SYSTEM_PROMPT = """You are an expert prompt generator for AI image synthesis (ComfyUI, Stable Diffusion, DALL-E).

Task:
Convert the user's description into a detailed, comma-separated prompt with visual descriptors and style guidance.

Style:
- Be specific and descriptive (NOT generic)
- Include character details: clothing, pose, expression, hair, build
- Include environment: lighting, setting, atmosphere, weather
- Include technical/style tags: art style, quality descriptors, camera angles
- Use natural language mixed with tag-like terms
- Make it vivid and paint a clear visual picture

Output format:
- Single comma-separated list (no numbering, bullets, or code blocks)
- Aim for 15-40 descriptive terms/phrases
- Lowercase, use underscores for multi-word phrases when natural
- Order by importance: subject → appearance → action → setting → style

DO NOT WORRY ABOUT:
- Using "official" tag names (Danbooru, etc.)
- Perfect grammar - natural descriptive phrases are fine
- Too many adjectives - more is better for image generation

GOOD examples:
- "1girl, long black hair, red eyes, maid outfit, standing, indoors, smiling, soft lighting, detailed face, high quality"
- "portrait, woman with flowing blonde hair, ethereal, glowing aura, fantasy setting, mystical atmosphere, dramatic lighting, highly detailed"
- "landscape, dense forest, misty morning, sunlight filtering through trees, detailed foliage, depth of field, cinematic"

Example input: "nun with lewd expression, adult content"
Example output: nun, religious robes, prayer beads, cathedral setting, soft candlelight, intimate pose, detailed features, artistic, sensual expression"""
    
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.danbooru_tags = self._load_danbooru_whitelist()
    
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
                system=self.SYSTEM_PROMPT,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 96,
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
        """Extract and validate tags from raw LLM response.
        
        Tags are parsed as-is without Danbooru whitelist validation.
        This allows for free-form descriptive prompts suitable for image generation.
        """
        raw_response = self._clean_response_text(raw_response)
        tags = []
        seen = set()
        for chunk in re.split(r"[\n,\.]+", raw_response):
            line = chunk.strip()
            if not line:
                continue
            # Remove common numbering/bullet patterns
            line = re.sub(r"^[\d\-\*\.\)]+\s*", "", line)
            line = line.strip().strip(".")
            if not line or any(c in line for c in [":", "=", ">"]):
                continue
            normalized = line.lower()
            if normalized in seen:
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
