"""Batch test: run 6 prompts across safe/creative/mature modes."""
from __future__ import annotations
import sys, time, json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.description_tagger import get_description_tagger

NON_EXPLICIT = [
    "a knight standing in the rain",
    "a catgirl baking cookies in a cozy kitchen",
    "a witch flying through a storm",
]

EXPLICIT = [
    "a maid getting groped on the train",
    "two elves kissing in a forest clearing",
    "a succubus seducing a priest",
]

ALL_PROMPTS = NON_EXPLICIT + EXPLICIT
MODES = ["safe", "creative", "mature"]

def main() -> None:
    tagger = get_description_tagger(
        model="richardyoung/qwen3-14b-abliterated:latest",
        post_count_threshold=500,
    )

    if not tagger.check_connection():
        print("ERROR: Cannot connect to Ollama. Is it running?")
        sys.exit(1)
    print(f"Connected to Ollama. Model: {tagger.model}\n")
    
    # Quick test with 2 prompts
    test_prompts = [
        "a knight standing in the rain",
        "a maid getting groped on the train",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        for mode in MODES:
            print(f"\n[{mode.upper()}]")
            start = time.time()
            try:
                res = tagger.generate_tags(prompt, creativity=mode)
                elapsed = time.time() - start
                tag_str = ", ".join(res.tags)
                print(f"  ✓ {len(res.tags)} tags in {elapsed:.1f}s")
                print(f"  Tags: {tag_str}")
            except Exception as e:
                elapsed = time.time() - start
                print(f"  ✗ ERROR ({elapsed:.1f}s): {e}")

if __name__ == "__main__":
    main()
