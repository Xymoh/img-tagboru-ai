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
    
    SYSTEM_PROMPT = """You are an expert Danbooru tagger.

Task:
Convert the user's description into precise real Danbooru-style tags.

Output format:
- Return a single comma-separated list of tags
- No numbering, no bullets, no code fences, no explanations
- Use lowercase only
- Use underscores for multi-word tags like black_hair or long_skirt
- Return at most 40 tags
- Prefer the most important tags first

Tagging rules:
- ONLY use real, established Danbooru tags - avoid inventing new tag names
- Prefer concrete visual facts over vague adjectives
- Include hair, eyes, clothing, pose, setting, species, and composition when present
- Include relevant style or quality tags only when supported by the description
- Include safety/content tags only when clearly implied by the description
- Avoid duplicates, filler words, and paraphrases of the same concept
- Aim for roughly 15-35 tags depending on how detailed the description is

Example:
girl, long_black_hair, red_eyes, maid_outfit, standing, indoors, looking_at_viewer"""
    
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
        
        Only tags that exist in the Danbooru whitelist are kept.
        Invalid/hallucinated tags are silently filtered out.
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
            
            # Only keep tags that are in the Danbooru whitelist
            if self.danbooru_tags and normalized not in self.danbooru_tags:
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
