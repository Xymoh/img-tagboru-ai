from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

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
    
    SYSTEM_PROMPT = """You are an expert Danbooru tagger. Given a text description, generate a list of appropriate Danbooru tags.

Rules:
- Output ONLY tags, one per line, no numbering or formatting
- Use underscores for multi-word tags (e.g., "black_hair", "long_skirt")
- Include character descriptions (hair color, clothing, body type)
- Include setting/scene tags if described
- Include style/art tags if relevant
- Include content warnings (nsfw, explicit) if described
- Keep tags lowercase
- Do NOT include explanations or commentary, only tags
- Aim for 20-40 tags depending on description complexity"""
    
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_available_models(self) -> list[str]:
        """Get list of available local models."""
        try:
            response = self.client.list()
            # Response.models is a list of Model objects with .model attribute
            return [m.model for m in response.models]
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
    
    @staticmethod
    def _parse_tags(raw_response: str) -> list[str]:
        """Extract tags from raw LLM response."""
        tags = []
        for line in raw_response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove common numbering/bullet patterns
            line = re.sub(r"^[\d\-\*\.\)]+\s*", "", line)
            line = line.strip()
            if line and not any(c in line for c in [":", "=", ">"]):
                tags.append(line.lower())
        return tags


def get_description_tagger(
    host: str = DescriptionTagger.DEFAULT_HOST,
    model: str = DescriptionTagger.DEFAULT_MODEL,
) -> DescriptionTagger:
    """Factory function to get a description tagger instance."""
    return DescriptionTagger(host=host, model=model)
