from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:
    ollama = None

from backend.tag_index import TagFrequencyIndex, get_tag_index

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tags considered "actor/act" for the action-only copy feature
_ACTOR_ACT_TAGS: set[str] = {
    "1girl", "2girls", "3girls", "multiple_girls",
    "1boy", "2boys", "3boys", "multiple_boys",
    "fellatio", "blowjob", "oral", "deepthroat", "cunnilingus",
    "sex", "vaginal", "anal", "anal_sex", "paizuri", "handjob", "footjob",
    "doggystyle", "missionary", "cowgirl_position", "reverse_cowgirl",
    "sex_from_behind", "riding", "straddling",
    "gangbang", "group_sex", "rape", "forced", "double_penetration",
    "69", "masturbation", "fingering", "fisting",
    "squirting", "orgasm",
    "kneeling", "bent_over", "on_all_fours", "on_back", "legs_up",
    "spread_legs", "spread_pussy", "lying",
    "penis", "erection", "vagina", "pussy", "clitoris",
    "breasts", "large_breasts", "bare_breasts", "nipples",
    "ass", "bare_ass", "spread_ass",
    "open_mouth", "tongue_out", "saliva", "drooling", "gagging",
    "penis_in_mouth", "cum_in_mouth",
    "cum", "cumshot", "cum_on_face", "cum_on_body", "creampie", "cum_inside",
    "facial",
    "ahegao", "moaning", "panting", "blush", "tears",
    "submission", "dominance", "power_dynamic", "restraints", "bondage",
}

# ---------------------------------------------------------------------------
# Semantic deduplication map
#   canonical_tag → [synonyms that should be collapsed into canonical]
# ---------------------------------------------------------------------------

_SEMANTIC_DEDUP_MAP: dict[str, list[str]] = {
    "fellatio":         ["blowjob", "oral", "bj", "sucking_dick", "deepthroat"],
    "cum_in_mouth":     ["cum_on_lips", "cum_on_tongue", "cum_in_throat"],
    "facial":           ["cum_on_face", "cum_on_hair", "cum_on_forehead"],
    "cum_on_body":      ["cum_on_chest", "cum_on_stomach", "cum_on_hands",
                         "cum_on_feet", "cumshot_on_body"],
    "night":            ["darkness", "dim_lighting", "dark", "low_light", "dark_ambience"],
    "alley":            ["alleyway", "back_alley"],
    "moaning":          ["panting", "heavy_breathing"],
    "tongue_out":       ["tongue"],
    "erection":         ["erect_penis", "hard"],
    "street":           ["road", "sidewalk", "pavement", "cobblestone"],
    "urban":            ["city", "cityscape", "downtown"],
    "saliva":           ["drooling", "drool", "wet"],
    "kneeling":         ["on_knees", "kneel"],
    "nun":              ["nun_outfit", "religious_sister", "nun_headdress", "traditional_nun"],
}

# Build reverse index for fast lookup: synonym → canonical
_SYNONYM_TO_CANONICAL: dict[str, str] = {}
for _canon, _syns in _SEMANTIC_DEDUP_MAP.items():
    for _s in _syns:
        _SYNONYM_TO_CANONICAL[_s] = _canon


# ---------------------------------------------------------------------------
# Few-shot examples per mode
# ---------------------------------------------------------------------------

_FEWSHOT_EXAMPLES: dict[str, list[tuple[str, str]]] = {
    "safe": [
        (
            "cat girl sitting on a bench in the park",
            "1girl, cat_ears, cat_tail, sitting, bench, park, outdoors, day, sky, tree, smile, short_hair, blue_skirt",
        ),
        (
            "warrior with a sword in a forest",
            "1girl, warrior, holding_sword, sword, armor, forest, outdoors, day, standing, determined, wind, cape",
        ),
        (
            "a nun giving a blowjob in the alley",
            "1girl, 1boy, nun, veil, cross, alley, night, street, kneeling, blush, shadow, looking_at_viewer, outdoors, robe",
        ),
    ],
    "creative": [
        (
            "a nun in the alley",
            "1girl, nun, veil, cross, alley, night, street, standing, looking_at_viewer, shadow, brick_wall, outdoors, moonlight",
        ),
        (
            "cat girl sitting on a bench in the park",
            "1girl, cat_ears, cat_tail, sitting, bench, park, outdoors, day, sky, tree, smile, short_hair, blue_skirt, sunlight, cherry_blossoms",
        ),
        (
            "warrior with a sword in a forest",
            "1girl, warrior, holding_sword, sword, armor, forest, outdoors, day, standing, determined, wind, cape, dappled_light, depth_of_field",
        ),
        (
            "a nun giving a blowjob in the alley",
            "1girl, 1boy, nun, veil, cross, fellatio, kneeling, alley, night, street, open_mouth, penis, blush, saliva",
        ),
    ],
    "mature": [
        (
            "a nun in the alley",
            "1girl, nun, veil, cross, alley, night, street, skirt_lift, showing_panties, blush, naughty_face, leaning_forward, shadow, dim_lighting, outdoors, moonlight",
        ),
        (
            "a nun giving a blowjob in the alley",
            "1girl, 1boy, nun, veil, cross, fellatio, kneeling, alley, night, street, open_mouth, penis, erection, saliva, blush, eye_contact",
        ),
        (
            "woman fucked from behind in bedroom",
            "1girl, 1boy, doggystyle, sex_from_behind, bent_over, ass, penis, penetration, nude, blush, bed, bedroom, indoors, night",
        ),
    ],
}


# ---------------------------------------------------------------------------
# DescriptionTagResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptionTagResult:
    tags: list[str]
    raw_response: str
    model: str
    actor_tags: list[str] = None
    scene_tags: list[str] = None

    def __post_init__(self) -> None:
        if self.actor_tags is None:
            actor = [t for t in self.tags if t in _ACTOR_ACT_TAGS]
            scene = [t for t in self.tags if t not in _ACTOR_ACT_TAGS]
            object.__setattr__(self, "actor_tags", actor)
            object.__setattr__(self, "scene_tags", scene)


# ---------------------------------------------------------------------------
# DescriptionTagger
# ---------------------------------------------------------------------------


class DescriptionTagger:
    """Generate Danbooru tags from text descriptions using local LLM (Ollama).

    Uses a TagFrequencyIndex backed by danbooru_tags_post_count.csv to:
      - Build vocabulary-grounded system prompts (LLM prefers common tags)
      - Post-process LLM output (strip rare tags, deduplicate synonyms)
    """

    DEFAULT_MODEL = "richardyoung/qwen3-14b-abliterated:latest"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_CREATIVITY = "creative"
    MAX_TAGS = 50  # Increased for better coverage
    _NSFW_MODES = {"mature"}
    _VALID_CREATIVITIES = {"safe", "creative", "mature"}
    
    # Model-specific prefill tokens for reasoning models
    _MODEL_PREFILLS = {
        "qwen": "<think>",
        "gemma": "<|channel>thought",
        "deepseek": None,  # DeepSeek-V4-Flash works better without prefill
    }

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        post_count_threshold: int = 500,
    ) -> None:
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
        self.tag_index = get_tag_index(min_threshold=post_count_threshold)
        self._system_prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Post-count threshold accessor
    # ------------------------------------------------------------------

    @property
    def post_count_threshold(self) -> int:
        return self.tag_index.min_threshold

    def set_post_count_threshold(self, value: int) -> None:
        self.tag_index.set_min_threshold(value)
        self._system_prompt_cache.clear()

    # ------------------------------------------------------------------
    # System prompt builders  (vocabulary-grounded)
    # ------------------------------------------------------------------

    _VOCAB_CATEGORIES_BY_MODE: dict[str, list[str]] = {
        "safe":     ["character", "clothing", "setting", "action", "style"],
        "creative": ["character", "clothing", "setting", "action", "style", "supernatural"],
        "mature":   ["character", "clothing", "setting", "action", "style", "supernatural", "explicit"],
    }

    _VOCAB_SIZE_BY_MODE: dict[str, int] = {
        "safe": 8,
        "creative": 10,
        "mature": 10,
    }

    # Tags that are franchise/game proper-noun collisions — they match generic
    # keywords by accident (e.g. "rain" matching "master_detective_archives:_rain_code",
    # "knight" matching "hollow_knight"). These are never shown in vocabulary sections.
    _BLOCKED_VOCAB_TAGS: set[str] = {
        "hollow_knight", "knight_(hollow_knight)", "flower_knight_girl",
        "master_detective_archives:_rain_code",
        "shachiku_succubus_no_hanashi", "high_priest_(ragnarok_online)",
        "priest_(ragnarok_online)", "king", "queen", "chen",
        "standing_sex", "standing_split", "meta_knight", "standing_cunnilingus",
        "lord_knight_(ragnarok_online)", "little_witch_academia", "sue_storm",
        "storm_(x-men)", "flying_kick", "flying_sweatdrops", "priest_(dq3)",
        "male_priest_(dungeon_and_fighter)", "disgaea",
        "knight_(chess)", "standing_missionary", "standing_on_liquid",
        "ranni_the_witch", "sanoba_witch", "flying_fish",
        "cleaning_&_clearing_(blue_archive)", "kissing_penis",
        "kissing_cheek", "kissing_forehead", "kissing_hand",
        "rain_world", "rain_code",
        "flying_teardrops", "flying_paper",
    }

    # Words to skip when extracting keywords from user descriptions
    _STOP_WORDS: set[str] = {
        "a", "an", "the", "is", "at", "with", "and", "being", "by", "in", "on",
        "of", "to", "for", "it", "as", "or", "its", "from", "that", "this",
        "was", "are", "were", "been", "has", "had", "have", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "shall",
        "not", "no", "but", "if", "then", "else", "when", "where", "who",
        "how", "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "so", "than", "too", "very",
        "just", "about", "above", "after", "again", "against", "below",
        "between", "during", "into", "through", "under", "up", "down", "out",
        "off", "over", "while", "her", "his", "she", "he", "they", "them",
        "me", "my", "we", "our", "you", "your", "i", "him", "us", "their",
        "giving", "give", "gets", "get", "got", "make", "made", "making",
        "like", "look", "looking", "see", "seen", "want", "using", "use",
        "take", "taken", "taking", "come", "came", "go", "going", "went",
        "there", "here", "now", "then", "also", "still", "even", "much",
        "many", "any", "one", "two", "three", "well", "back", "way",
    }

    @staticmethod
    def _extract_keywords(description: str) -> list[str]:
        """Extract meaningful lowercase keywords from a user description."""
        import re
        # Lowercase and split on non-alphanumeric
        words = re.findall(r"[a-z0-9]+", description.lower())
        # Filter: at least 3 chars, no pure digits, no stop words
        keywords: list[str] = []
        seen: set[str] = set()
        for w in words:
            if len(w) < 3:
                continue
            if w.isdigit():
                continue
            if w in DescriptionTagger._STOP_WORDS:
                continue
            if w in seen:
                continue
            seen.add(w)
            keywords.append(w)
        return keywords

    def _build_vocabulary_section(
        self, creativity: str, description: str, exclude: set[str] | None = None,
    ) -> str:
        """Build the AVAILABLE VOCABULARY block, including prompt-specific tags."""
        categories = self._VOCAB_CATEGORIES_BY_MODE.get(
            creativity, self._VOCAB_CATEGORIES_BY_MODE["creative"]
        )
        n = self._VOCAB_SIZE_BY_MODE.get(creativity, 12)
        cat_tags = self.tag_index.top_by_category(
            n=n, categories=categories,
            exclude=(exclude or set()) | self._BLOCKED_VOCAB_TAGS,
        )

        label_map = {
            "character": "Character / Participants",
            "clothing":  "Clothing / Accessories",
            "setting":   "Setting / Environment",
            "action":    "Action / Pose / Expression",
            "style":     "Style / Quality",
            "supernatural": "Supernatural / Horror / Monster",
            "explicit":  "Explicit / Adult",
        }

        lines: list[str] = []
        for cat in categories:
            tags = cat_tags.get(cat, [])
            if tags:
                label = label_map.get(cat, cat.title())
                lines.append(f"- {label}: {', '.join(tags[:n])}")

        # --- Prompt-specific: inject tags matching user's keywords ---
        keywords = self._extract_keywords(description)
        if keywords:
            kw_tags = self.tag_index.search_by_keywords(
                keywords,
                n=5,
                min_count=100,
                exclude=(exclude or set()) | self._BLOCKED_VOCAB_TAGS,
            )
            # Filter out tags already shown in category sections
            shown: set[str] = set()
            for line in lines:
                for tag in line.split(", "):
                    if tag:
                        shown.add(tag.rstrip("."))
            # Collect up to 4 tags per keyword to ensure niche words surface
            per_kw: dict[str, list[str]] = {}
            for tag in kw_tags:
                if tag in shown or tag in self._BLOCKED_VOCAB_TAGS:
                    continue
                for kw in keywords:
                    kw_l = kw.lower()
                    tag_parts = tag.split("_")
                    if kw_l in tag_parts or (len(tag) >= 4 and tag in kw_l):
                        per_kw.setdefault(kw_l, []).append(tag)
                        break
            # Interleave: take 1st from each keyword, then 2nd, etc.
            interleaved: list[str] = []
            seen_kw: set[str] = set()
            for i in range(3):  # up to 3 per keyword
                for kw in keywords:
                    kw_l = kw.lower()
                    bucket = per_kw.get(kw_l, [])
                    if i < len(bucket):
                        tag = bucket[i]
                        if tag not in seen_kw:
                            seen_kw.add(tag)
                            interleaved.append(tag)
                        if len(interleaved) >= 20:
                            break
                if len(interleaved) >= 20:
                    break
            if interleaved:
                lines.append(
                    f"- RELEVANT TO YOUR DESCRIPTION: {', '.join(interleaved)}"
                )

        return "\n".join(lines)

    _MODE_EXTRA_RULES: dict[str, str] = {
        "safe": (
            "SAFE MODE RULES:\n"
            "- NEVER use explicit/sexual tags (penis, fellatio, sex, nude, cum, erection, etc.).\n"
            "- If the description mentions sexual acts, describe ONLY the scene context:\n"
            "  the characters' appearance, clothing, setting, mood, lighting, and non-sexual pose.\n"
            "- Focus on atmosphere, environment, and character details.\n"
            "- Output 15-25 tags covering: participants, clothing, setting, lighting, mood."
        ),
        "creative": (
            "CREATIVE MODE RULES:\n"
            "- You may use mild action/pose tags implied by the description.\n"
            "- Add scene context, lighting, and composition cues where appropriate.\n"
            "- Do NOT add cum/ejaculation/finish tags unless the description explicitly\n"
            "  mentions cum, creampie, ejaculation, or similar finish-related words.\n"
            "- Output 20-35 tags covering: participants, action, clothing, setting, lighting, mood, style."
        ),
        "mature": (
            "MATURE MODE RULES:\n"
            "- You may use explicit/sexual tags freely when the description calls for them.\n"
            "- For explicit scenes: include participant counts (1girl, 1boy, etc.), sexual acts\n"
            "  (fellatio, sex, penetration, etc.), body parts (penis, breasts, pussy, etc.),\n"
            "  expressions (ahegao, blush, open_mouth), and fluids (cum, saliva, etc.).\n"
            "- For non-explicit descriptions, be BOLD and SUGGESTIVE: add clothing-state\n"
            "  tags (clothes_lift, skirt_lift, showing_panties, underwear, torn_clothes),\n"
            "  body language (bent_over, leaning_forward, legs_apart, arched_back),\n"
            "  expressions (blush, naughty_face, bedroom_eyes, licking_lips), and mood\n"
            "  (dim_lighting, night, intimate, shadow, candlelight).\n"
            "- Be inventive with atmosphere and implication. Lean into the adult reading\n"
            "  of the scene without fabricating sex acts that weren't mentioned.\n"
            "- Include participant counts, clothing details, expressions, and setting.\n"
            "- Do NOT add cum/ejaculation/finish tags unless the description explicitly\n"
            "  mentions cum, creampie, ejaculation, or similar finish-related words.\n"
            "- Output 25-40 tags covering: participants, explicit acts, body parts, clothing state,\n"
            "  expressions, fluids, setting, lighting, mood."
        ),
    }

    def _build_system_prompt(self, creativity: str, description: str = "") -> str:
        """Build a vocabulary-grounded system prompt for *creativity* mode.

        When *description* is provided, prompt-specific tags matching the
        description keywords are injected into the vocabulary section.
        """
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        vocab = self._build_vocabulary_section(creativity, description)
        extra_rules = self._MODE_EXTRA_RULES.get(creativity, "")

        prompt = f"""You are a Danbooru-style image tag generator. Convert the user's description
into a comma-separated list of tags.

CRITICAL RULES:
- Output ONLY tags separated by commas. No prose, no markdown, no explanations.
- Do NOT write <think> or reasoning blocks. Output tags immediately.
- Use lowercase with underscores for spaces (Danbooru format).
- Prefer tags from the AVAILABLE VOCABULARY below — these are the most commonly used
  tags and will produce the strongest image generation results.
- Only add tags that are DIRECTLY implied by the description. Do NOT invent elements
  that are not mentioned or strongly suggested.
- Limit output to the natural number of tags the description warrants.
{extra_rules}

AVAILABLE VOCABULARY (most common Danbooru tags — prefer these):
{vocab}

OUTPUT FORMAT (example):
1girl, solo, long_hair, blue_eyes, smile, outdoors, day, sky, cloud

Now output ONLY the comma-separated tags for the user's description."""

        return prompt

    # ------------------------------------------------------------------
    # Generation prompt builders
    # ------------------------------------------------------------------

    def _build_generation_prompt(
        self, description: str, creativity: str, attempt_index: int = 0
    ) -> str:
        """Build the user message sent to the LLM containing description + few-shot."""
        examples = _FEWSHOT_EXAMPLES.get(creativity, _FEWSHOT_EXAMPLES["creative"])

        # Deterministically pick 2-3 examples based on description hash
        seed = hashlib.md5(description.strip().lower().encode("utf-8")).hexdigest()
        idx = int(seed[:8], 16) % max(1, len(examples) - 2)
        selected = examples[idx : idx + 3] if len(examples) > 3 else examples

        parts: list[str] = []
        for inp, out in selected:
            parts.append(f'Input: "{inp}"\nOutput: {out}')
        parts.append(f'Input: "{description.strip()}"\nOutput:')

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tag parsing & post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response_text(raw_response: str) -> str:
        """Remove common wrapper formats from model output before parsing."""
        text = raw_response.strip()
        text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")
        text = re.sub(r"\\(?:text|mathrm|mathbf|mathtt|operatorname)\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
        text = text.replace("{", " ").replace("}", " ")
        return text.strip()

    def _parse_tags(self, raw_response: str) -> list[str]:
        """Extract tags from LLM response, validating against the CSV index.

        Strips  think blocks then parses comma-separated tag output.
        If the result is sparse (LLM cut off mid-reasoning), the caller
        supplements with CSV keyword search results — a cleaner fallback
        than mining prose for individual words.
        """
        text = re.sub(r"<think>.*?</think>", " ", raw_response, flags=re.DOTALL)
        text = re.sub(r"<think>.*", " ", text, flags=re.DOTALL)
        text = self._clean_response_text(text)

        tags: list[str] = []
        seen: set[str] = set()

        for chunk in re.split(r"[\n,]+", text):
            line = chunk.strip()
            if not line:
                continue
            line = re.sub(r"^[0-9]+[.)]\s*|^[\-\*]\s+", "", line)
            line = line.strip().strip(".")
            if not line or any(c in line for c in [":", "=", ">"]):
                continue
            normalized = line.lower().replace(" ", "_")
            if normalized in seen:
                continue
            if not self.tag_index.is_valid(normalized):
                continue
            if normalized in self._BLOCKED_VOCAB_TAGS:
                continue
            seen.add(normalized)
            tags.append(normalized)
            if len(tags) >= self.MAX_TAGS:
                break

        return tags

    # Tags that should ONLY appear when the description explicitly mentions them.
    # These are blocked in post-processing to prevent LLM hallucination.
    _SENSITIVE_TAGS: set[str] = {
        "loli", "child", "trap", "futanari", "shota",
        "rape", "tentacles", "bestiality", "guro", "vore",
        "necrophilia", "scat", "watersports", "urination",
    }

    def _post_process_tags(self, tags: list[str], creativity: str, target_count: int,
                           description: str = "") -> list[str]:
        """Six-step post-processing pipeline.

        1. Validate against CSV whitelist (already done by _parse_tags)
        2. Filter by post_count threshold
        3. Block sensitive tags unless explicitly mentioned in description
        4. Semantic deduplication
        5. Sort by post_count descending
        6. Cap to target count
        """
        # Step 2: threshold filter
        tags = self.tag_index.filter_by_threshold(tags)

        # Step 3: block sensitive tags unless description explicitly mentions them
        if description:
            desc_lower = description.lower()
            tags = [t for t in tags if t not in self._SENSITIVE_TAGS
                    or any(kw in desc_lower for kw in (t, t.replace("_", " ")))]

        # Step 4: semantic dedup — keep highest-count tag per synonym group
        seen_canonicals: set[str] = set()
        seen_synonyms: set[str] = set()
        deduped: list[str] = []
        for tag in tags:
            if tag in seen_synonyms:
                continue
            canon = _SYNONYM_TO_CANONICAL.get(tag)
            if canon is not None:
                # This tag is a synonym — skip if canonical already included
                if canon in seen_canonicals:
                    seen_synonyms.add(tag)
                    continue
                # Replace with canonical if canonical also in tags
                if canon in tags:
                    seen_synonyms.add(tag)
                    continue
            # Check if tag is a canonical with synonyms already present
            if tag in _SEMANTIC_DEDUP_MAP:
                if tag in seen_canonicals:
                    continue
                seen_canonicals.add(tag)
                for syn in _SEMANTIC_DEDUP_MAP[tag]:
                    seen_synonyms.add(syn)
            deduped.append(tag)

        # Step 4: sort by post_count descending
        deduped = self.tag_index.sort_by_count(deduped)

        # Step 5: cap
        return deduped[:target_count]

    # ------------------------------------------------------------------
    # Literal / actor extraction  (kept from old code — still useful)
    # ------------------------------------------------------------------

    def _extract_literal_tags_from_description(self, description: str) -> list[str]:
        """Map explicit user words/phrases to known tags for safe mode."""
        desc = description.lower()
        phrase_map = [
            ("1girl", "1girl"),
            ("1boy", "1boy"),
            ("2boys", "2boys"),
            ("2girls", "2girls"),
            ("fellatio", "fellatio"),
            ("oral", "oral"),
            ("sex", "sex"),
            ("male", "1boy"),
            ("female", "1girl"),
            ("fat", "fat"),
            ("chubby", "fat"),
            ("dark skinned", "dark_skin"),
            ("dark-skinned", "dark_skin"),
            ("dark skin", "dark_skin"),
            ("blonde", "blonde_hair"),
            ("blond", "blonde_hair"),
            ("blonde hair", "blonde_hair"),
            ("blonder", "blonde_hair"),
            ("blonde-haired", "blonde_hair"),
            ("nun", "nun"),
            ("nude", "nude"),
            ("cum", "cum"),
            ("sloppy", "open_mouth"),
            ("messy", "messy_hair"),
        ]
        tags: list[str] = []
        seen: set[str] = set()
        for phrase, tag in phrase_map:
            if phrase not in desc:
                continue
            if not self.tag_index.is_valid(tag):
                continue
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _extract_actor_tags_from_description(self, description: str) -> list[str]:
        """Infer participant tags so visible actors are always included."""
        desc = description.lower()
        tags: list[str] = []

        def add(tag: str) -> None:
            if not self.tag_index.is_valid(tag):
                return
            if tag not in tags:
                tags.append(tag)

        male_markers = ["man", "male", "boy", "guy", "husband", "father", "priest"]
        female_markers = ["woman", "female", "girl", "nun", "lady", "wife", "mother", "sister"]
        plural_male_markers = ["2 men", "two men", "multiple men", "several men", "boys", "men"]
        plural_female_markers = ["2 women", "two women", "multiple women", "several women", "girls", "women"]

        has_male = any(marker in desc for marker in male_markers)
        has_female = any(marker in desc for marker in female_markers)

        if any(marker in desc for marker in plural_male_markers):
            add("2boys")
        elif has_male:
            add("1boy")

        if any(marker in desc for marker in plural_female_markers):
            add("2girls")
        elif has_female:
            add("1girl")

        # Implicit partner: acts that require a partner even if not mentioned
        partner_acts = [
            "blowjob", "fellatio", "sex", "doggystyle", "missionary",
            "cowgirl", "paizuri", "handjob", "footjob", "cunnilingus", "anal",
            "fuck", "penetration", "intercourse",
        ]
        multi_partner_acts = [
            "spitroast", "gangbang", "double_penetration", "orgy",
            "threesome", "group_sex", "multiple men",
        ]
        has_partner = any(act in desc for act in partner_acts)
        has_multi = any(act in desc for act in multi_partner_acts)
        if has_multi and not tags:
            add("1girl")
            add("multiple_boys")
        elif has_multi and len(tags) == 1:
            add("multiple_boys" if tags[0] == "1girl" else "1girl")
        elif has_partner and not tags:
            add("1girl")
            add("1boy")
        elif has_partner and len(tags) == 1:
            add("1girl" if tags[0] == "1boy" else "1boy")

        return tags

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    def generate_tags(
        self, description: str, creativity: str = DEFAULT_CREATIVITY
    ) -> DescriptionTagResult:
        """Generate Danbooru tags from a text description.

        Uses the TagFrequencyIndex for vocabulary-grounded prompting and
        post-processing quality filtering.  Automatically retries once
        when the LLM returns fewer than 3 parseable tags (a common failure
        mode with abliterated models whose <think> blocks consume the
        num_predict budget).
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        creativity = (creativity or self.DEFAULT_CREATIVITY).strip().lower()
        if creativity not in self._VALID_CREATIVITIES:
            creativity = self.DEFAULT_CREATIVITY

        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )

        mode_options = {
            "safe":     {"base_temp": 0.50, "num_predict": 300, "target_tags": 20, "top_p": 0.92},
            "creative": {"base_temp": 0.65, "num_predict": 450, "target_tags": 30, "top_p": 0.95},
            "mature":   {"base_temp": 1.05, "num_predict": 600, "target_tags": 40, "top_p": 0.98},
        }
        opts = mode_options[creativity]
        target_tags = opts["target_tags"]

        # Pre-compute literal/actor tags (stable across retries)
        literal_tags = self._extract_literal_tags_from_description(description)
        actor_tags = self._extract_actor_tags_from_description(description)

        last_error: Exception | None = None
        raw_text_final: str = ""

        # Detect model family for prefill optimization
        model_lower = self.model.lower()
        prefill_token = None
        if "qwen" in model_lower:
            prefill_token = self._MODEL_PREFILLS.get("qwen")
        elif "gemma" in model_lower:
            prefill_token = self._MODEL_PREFILLS.get("gemma")
        elif "deepseek" in model_lower:
            prefill_token = self._MODEL_PREFILLS.get("deepseek")

        for attempt in range(2):
            try:
                # Build prompts
                system_prompt = self._build_system_prompt(creativity, description)
                gen_prompt = self._build_generation_prompt(description, creativity)
                
                # Add prefill if model supports it
                if prefill_token:
                    gen_prompt = f"{gen_prompt}\n{prefill_token}"

                # Bump temperature on retry to shake the model out of a rut
                temp = opts["base_temp"]
                if attempt > 0:
                    temp = min(1.35, temp + 0.20)

                response = self.client.generate(
                    model=self.model,
                    prompt=gen_prompt,
                    system=system_prompt,
                    options={
                        "temperature": temp,
                        "top_p": opts["top_p"],
                        "top_k": 50,
                        "repeat_penalty": 1.12,
                        "num_predict": opts["num_predict"],
                        "stop": ["</think>", "\n\n\n", "\n\nInput:", "Output:", "<think>"],  # Aggressive stop
                    },
                    stream=False,
                )
                raw_text = response.get("response", "").strip()
                raw_text_final = raw_text

                # Parse and post-process
                parsed_tags = self._parse_tags(raw_text)

                # Inject literal/actor tags at front
                injected: set[str] = set()
                merged: list[str] = []
                for tag in actor_tags + literal_tags + parsed_tags:
                    if tag in injected:
                        continue
                    injected.add(tag)
                    merged.append(tag)

                final_tags = self._post_process_tags(
                    merged, creativity, target_tags, description
                )

                # Only accept if we got a meaningful result
                if len(final_tags) >= 3:
                    return DescriptionTagResult(
                        tags=final_tags[:target_tags],
                        raw_response=raw_text_final,
                        model=self.model,
                    )
                # Otherwise loop to retry

            except Exception as e:
                last_error = e
                # Don't retry on connection/network errors
                if "connect" in str(e).lower() or "refused" in str(e).lower():
                    raise RuntimeError(f"Tag generation failed: {e}")

        # Exhausted retries — return whatever we have (even if sparse)
        # or raise if every attempt was an exception
        if last_error is not None:
            raise RuntimeError(f"Tag generation failed after retries: {last_error}")

        # Rebuild best-effort result from the last attempt
        parsed_tags = self._parse_tags(raw_text_final)
        injected = set()
        merged = []
        for tag in actor_tags + literal_tags + parsed_tags:
            if tag in injected:
                continue
            injected.add(tag)
            merged.append(tag)
        final_tags = self._post_process_tags(merged, creativity, target_tags, description)

        return DescriptionTagResult(
            tags=final_tags[:target_tags],
            raw_response=raw_text_final,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Ollama connection & model management
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=3) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """Get list of available local models."""
        try:
            request = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(request, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            models = []
            for item in payload.get("models", []):
                name = item.get("model") or item.get("name")
                if name:
                    models.append(name)
            return models
        except Exception:
            try:
                resp = self.client.list()
                return [m.model for m in resp.models if getattr(m, "model", None)]
            except Exception:
                return []

    def pull_model(self, model: str) -> None:
        """Download a model from Ollama registry."""
        try:
            self.client.pull(model)
        except Exception as e:
            raise RuntimeError(f"Failed to pull model {model}: {e}")

    def delete_model(self, model: str) -> None:
        """Delete a model from the local Ollama instance."""
        try:
            self.client.delete(model)
        except Exception as e:
            raise RuntimeError(f"Failed to delete model {model}: {e}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_description_tagger(
    host: str = DescriptionTagger.DEFAULT_HOST,
    model: str = DescriptionTagger.DEFAULT_MODEL,
    post_count_threshold: int = 500,
) -> DescriptionTagger:
    """Factory function to get a description tagger instance."""
    return DescriptionTagger(
        host=host,
        model=model,
        post_count_threshold=post_count_threshold,
    )
