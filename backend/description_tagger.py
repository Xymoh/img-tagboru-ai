from __future__ import annotations

import csv
import hashlib
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


# Tags considered "actor/act" for the action-only copy feature
_ACTOR_ACT_TAGS: set[str] = {
    # participant counts
    "1girl", "2girls", "3girls", "multiple_girls",
    "1boy", "2boys", "3boys", "multiple_boys",
    # explicit acts
    "fellatio", "blowjob", "oral", "deepthroat", "cunnilingus",
    "sex", "vaginal", "anal", "anal_sex", "paizuri", "handjob", "footjob",
    "doggystyle", "missionary", "cowgirl_position", "reverse_cowgirl",
    "sex_from_behind", "riding", "straddling",
    "gangbang", "group_sex", "rape", "forced", "double_penetration",
    "69", "masturbation", "fingering", "fisting",
    "squirting", "orgasm",
    # positions / poses
    "kneeling", "bent_over", "on_all_fours", "on_back", "legs_up",
    "spread_legs", "spread_pussy", "lying", "cowgirl",
    # body parts (explicit)
    "penis", "erection", "vagina", "pussy", "clitoris",
    "breasts", "large_breasts", "bare_breasts", "nipples",
    "ass", "bare_ass", "spread_ass",
    # oral state
    "open_mouth", "tongue_out", "saliva", "drooling", "gagging",
    "penis_in_mouth", "cum_in_mouth",
    # finish
    "cum", "cumshot", "cum_on_face", "cum_on_body", "creampie", "cum_inside",
    "facial",
    # expressions
    "ahegao", "moaning", "panting", "blush", "tears",
    # power dynamics
    "submission", "dominance", "power_dynamic", "restraints", "bondage",
}


@dataclass(frozen=True)
class DescriptionTagResult:
    tags: list[str]
    raw_response: str
    model: str
    # Tags split by category — populated by generate_tags()
    actor_tags: list[str] = None   # participants + acts + positions
    scene_tags: list[str] = None   # clothing, setting, atmosphere, style

    def __post_init__(self) -> None:
        # Compute splits lazily if not provided
        if self.actor_tags is None:
            actor = [t for t in self.tags if t in _ACTOR_ACT_TAGS]
            scene = [t for t in self.tags if t not in _ACTOR_ACT_TAGS]
            object.__setattr__(self, "actor_tags", actor)
            object.__setattr__(self, "scene_tags", scene)


class DescriptionTagger:
    """Generate Danbooru tags from text descriptions using local LLM (Ollama)."""
    
    DEFAULT_MODEL = "qwen2:7b"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_CREATIVITY = "creative"
    MAX_TAGS = 40
    # Modes that should bypass the SFW image-tagger vocabulary filter
    _NSFW_MODES = {"mature"}
    
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
        self.system_prompt = self._build_system_prompt(self.DEFAULT_CREATIVITY)
    
    def _load_known_tags(self) -> set[str]:
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
    
    def _build_system_prompt(self, creativity: str = DEFAULT_CREATIVITY) -> str:
        """Build a grounded system prompt with real tags from the trained model."""
        if not self.known_tags:
            # Fallback if no trained tags available
            return self._get_default_system_prompt(creativity)
        
        # Create tag sets by category with more examples
        character_tags = sorted([t for t in self.known_tags if any(x in t for x in ['girl', 'boy', '1girl', '2girls', '1boy', 'boy', 'hair', 'eyes'])])[:20]
        clothing_tags = sorted([t for t in self.known_tags if any(x in t for x in ['dress', 'shirt', 'skirt', 'coat', 'outfit', 'top', 'bottom', 'uniform'])])[:15]
        setting_tags = sorted([t for t in self.known_tags if any(x in t for x in ['indoors', 'outdoors', 'beach', 'forest', 'room', 'sky', 'bedroom', 'office'])])[:12]
        action_tags = sorted([t for t in self.known_tags if any(x in t for x in ['standing', 'sitting', 'lying', 'dancing', 'running', 'holding', 'reading', 'playing'])])[:12]
        style_tags = sorted([t for t in self.known_tags if any(x in t for x in ['painting', 'detailed', 'quality', 'lighting', 'style', 'realistic', 'sketch', 'watercolor'])])[:12]

        if creativity == "mature":
            return self._build_mature_system_prompt()

        mode_guidance = {
            "safe": "Stay literal. Only use tags directly implied by the input.",
            "creative": "Be inventive only within the description: add scene details, props, pose cues, or setting cues that are specifically suggested by the subject.",
            "extreme": "Push variety further, but every added tag must still be grounded in the description. Do not reuse generic atmosphere defaults.",
        }

        prompt = f"""You are a creative image-tag prompt generator.

    Mode guidance:
    - {mode_guidance.get(creativity, mode_guidance['creative'])}

    Goal:
    - Convert user description into a grounded comma-separated tag list.
    - Add scene details only when they fit the description.

    Critical constraints:
    - Use Danbooru-style tags (underscores for spaces, lowercase).
    - Output only tags separated by commas.
    - No prose, no bullets, no numbering.
    - Prefer a modest number of high-signal tags over filler.

    Tag style examples from trained vocabulary:
    - Character: {', '.join(character_tags[:10])}
    - Clothing: {', '.join(clothing_tags[:10])}
    - Setting: {', '.join(setting_tags[:10])}
    - Action: {', '.join(action_tags[:10])}
    - Style: {', '.join(style_tags[:10])}

    Output format:
    tag1, tag2, tag3, tag4, ..."""

        return prompt
    
    def _build_mature_system_prompt(self) -> str:
        """Dedicated system prompt for mature/explicit mode — Danbooru/Rule34 style."""
        return """You are an explicit adult image tag generator modeled on Danbooru and Rule34 tagging conventions.

Your job: convert any user description into a comma-separated list of explicit Danbooru-style tags.

RULES:
- Output ONLY a comma-separated tag list. No prose, no explanations, no markdown.
- Use Danbooru/Rule34 tag format: lowercase, underscores instead of spaces.
- Be explicit and graphic. This is adult content — do not soften or censor.
- Always include: participant count tags (1girl, 1boy, 2boys, etc.), explicit act tags, body part tags, position/pose tags, clothing/state tags.
- ORDER: participants → appearance/role → explicit acts → body parts → clothing/state → props/objects → setting → atmosphere.
- Capture ALL elements of the description, not just the sex act. If the user mentions occupation, appearance, props, setting, or context — include those tags.

CRITICAL — ACT CONSISTENCY:
- Identify the ONE primary sex act from the description (e.g. fellatio, doggystyle, missionary, gangbang).
- Use ONLY the body positions and body parts that naturally match that ONE act.
- Do NOT mix positions from different acts. A fellatio scene has kneeling, not bent_over or on_back or straddling.
- Fellatio → kneeling, open_mouth, penis, erection, saliva, eye_contact. NOT spread_legs, on_back, missionary.
- Doggystyle → bent_over, on_all_fours, sex_from_behind, ass. NOT kneeling or straddling.
- Missionary → on_back, legs_up, spread_legs. NOT bent_over or kneeling.
- Cowgirl → straddling, riding, cowgirl_position. NOT on_back or kneeling.

CONTEXT & APPEARANCE — Always capture these from the description:
- Roles/occupation: streamer, nurse, nun, teacher, student, maid, office_lady, police, school_uniform, etc.
- Appearance: ugly_man, old_man, muscular, chubby, dark_skin, pale_skin, tall, short, young, etc.
- Props/devices: phone, smartphone, webcam, camera, recording, microphone, headphones, computer, laptop
- Streaming/online: live_stream, streaming, camgirl, webcam, computer_screen
- Relationship: interracial, teacher/student, boss/employee, siblings, etc.
- Setting: indoors, outdoors, bedroom, office, bathroom, public, night, day
- Do NOT invent tags the user didn't hint at. Only add context tags that are DIRECTLY suggested by the description.

EXPLICIT TAG VOCABULARY (use freely and creatively):
- Acts: fellatio, cunnilingus, sex, rape, gangbang, doggystyle, missionary, 69, anal, paizuri, handjob, footjob, blowjob, creampie, cumshot, cum_in_mouth, cum_on_face, cum_on_body, squirting, orgasm, masturbation, fingering, fisting, double_penetration
- Body: penis, erection, vagina, pussy, clitoris, breasts, large_breasts, nipples, bare_breasts, ass, bare_ass, spread_legs, spread_pussy, open_mouth, tongue_out, ahegao, drooling, saliva
- Pose/state: nude, naked, topless, bottomless, partially_clothed, sex_from_behind, lying, kneeling, on_all_fours, bent_over, straddling, cowgirl_position, reverse_cowgirl, legs_up, legs_apart, on_back
- Emotion/expression: moaning, panting, blush, tears, ahegao, pleasure, lust
- Clothing (partial/erotic): lingerie, thighhighs, stockings, garter_belt, corset, bra, panties, bikini, see-through, torn_clothes, lifted_skirt, no_panties
- Extras: eye_contact, pov, from_behind, close-up, indoors, outdoors, bed, public_sex

Example input: "nun giving blowjob to priest"
Example output: 1girl, 1boy, nun, habit, cross, priest, church, fellatio, kneeling, eye_contact, open_mouth, penis, erection, saliva, blush, moaning, submission, power_dynamic, indoors

Example input: "woman fucked from behind"
Example output: 1girl, 1boy, doggystyle, sex_from_behind, bent_over, on_all_fours, ass, penis, penetration, moaning, nude, blush, panting

Example input: "streamer girl giving blowjob to ugly man while being recorded on phone"
Example output: 1girl, 1boy, streamer, ugly_man, interracial, fellatio, phone, recording, holding_phone, webcam, live_stream, kneeling, open_mouth, penis, erection, saliva, eye_contact, blush, nude, indoors

Output format: tag1, tag2, tag3, ..."""

    def _get_default_system_prompt(self, creativity: str = DEFAULT_CREATIVITY) -> str:
        """Fallback system prompt when trained tags aren't available."""
        if creativity == "mature":
            return self._build_mature_system_prompt()

        mode_guidance = {
            "safe": "Stay literal. Use only tags directly implied by the description.",
            "creative": "Add a few plausible scene details that are directly suggested by the description. Do not fall back to generic atmosphere tags.",
            "extreme": "Add a broader but still coherent set of scene details. Keep everything tied to the description.",
        }
        return f"""You are an expert AI art prompt generator.

Mode guidance:
- {mode_guidance.get(creativity, mode_guidance['creative'])}

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
    
    def generate_tags(self, description: str, creativity: str = DEFAULT_CREATIVITY) -> DescriptionTagResult:
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

        creativity = (creativity or DescriptionTagger.DEFAULT_CREATIVITY).strip().lower()
        if creativity not in {"safe", "creative", "extreme", "mature"}:
            creativity = DescriptionTagger.DEFAULT_CREATIVITY
        
        if not self.check_connection():
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
        
        try:
            prompts = [self._build_generation_prompt(description, creativity, 0)]
            if creativity == "extreme":
                prompts.append(self._build_generation_prompt(description, creativity, 1))

            mode_options = {
                "safe":     {"base_temp": 0.45, "temp_step": 0.06, "num_predict": 140, "target_tags": 16},
                "creative": {"base_temp": 0.58, "temp_step": 0.08, "num_predict": 180, "target_tags": 24},
                "extreme":  {"base_temp": 0.72, "temp_step": 0.10, "num_predict": 220, "target_tags": 32},
                "mature":   {"base_temp": 0.80, "temp_step": 0.10, "num_predict": 280, "target_tags": 35},
            }
            opts = mode_options[creativity]
            target_tags = opts["target_tags"]

            collected_tags: list[str] = []
            seen: set[str] = set()
            raw_responses: list[str] = []

            for idx, prompt in enumerate(prompts):
                extra_opts: dict = {}
                if creativity == "mature":
                    # Disable thinking/reasoning tokens for qwen3 models so output is pure tags
                    extra_opts["think"] = False

                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=self._build_system_prompt(creativity),
                    options={
                        "temperature": min(1.30, opts["base_temp"] + (idx * opts["temp_step"])),
                        "top_p": 0.95,
                        "top_k": 60,
                        "repeat_penalty": 1.10,
                        "num_predict": opts["num_predict"],
                    },
                    **extra_opts,
                    stream=False,
                )
                raw_text = response.get("response", "").strip()
                raw_responses.append(raw_text)

                for tag in self._parse_tags(raw_text, creativity):
                    if tag in seen:
                        continue
                    seen.add(tag)
                    collected_tags.append(tag)
                    # For mature mode keep collecting; don't stop early
                    if creativity != "mature" and len(collected_tags) >= target_tags:
                        break
                if creativity != "mature" and len(collected_tags) >= target_tags:
                    break

            if creativity == "safe":
                literal_tags = self._extract_literal_tags_from_description(description)
                merged: list[str] = []
                merged_seen: set[str] = set()
                for tag in literal_tags + collected_tags:
                    if tag in merged_seen:
                        continue
                    merged_seen.add(tag)
                    merged.append(tag)
                    if len(merged) >= target_tags:
                        break
                collected_tags = merged

            if creativity == "mature":
                actor_tags = self._extract_actor_tags_from_description(description)
                if actor_tags:
                    merged: list[str] = []
                    merged_seen: set[str] = set()
                    for tag in actor_tags + collected_tags:
                        if tag in merged_seen:
                            continue
                        merged_seen.add(tag)
                        merged.append(tag)
                        if len(merged) >= target_tags:
                            break
                    collected_tags = merged

            # If the model returned very few tags, ensure we include high-signal literal tags
            if len(collected_tags) < 3:
                literal_tags = self._extract_literal_tags_from_description(description)
                for tag in literal_tags:
                    if tag not in collected_tags:
                        collected_tags.insert(0, tag)
                    if len(collected_tags) >= max(3, min(target_tags, 6)):
                        break

            enriched_tags = self._enrich_tags_with_scene_ideas(
                description, collected_tags, creativity, target_tags
            )

            return DescriptionTagResult(
                tags=enriched_tags[:target_tags],
                raw_response="\n---\n".join(raw_responses),
                model=self.model,
            )
        except Exception as e:
            raise RuntimeError(f"Tag generation failed: {e}")

    def _build_generation_prompt(self, description: str, creativity: str, attempt_index: int) -> str:
        """Build a mode-aware prompt that keeps expansions grounded."""
        if creativity == "safe":
            return (
                f"{description}\n\n"
                "Use only literal tags directly implied by the description. "
                "Do not add cinematic, horror, atmosphere, lighting, or camera tags unless explicitly requested."
            )

        creative_focuses = [
            "Focus on the character, outfit, and pose. Add only one or two supporting tags that are directly suggested by the description.",
            "Focus on the emotional or devotional tone. Use tags that describe gesture, expression, and a single grounded context clue.",
            "Focus on small tangible details around the subject, not the whole scene. Prefer wardrobe, accessories, and hand details.",
        ]
        mature_focuses = [
            (
                "Generate explicit Danbooru/Rule34 tags for this scene. "
                "Capture EVERY element: participants (count), appearance/role (streamer, ugly_man, etc.), "
                "the explicit act, body parts, position, expression, clothing state, props (phone, webcam, etc.), "
                "setting, and atmosphere. Do NOT include any prose — output ONLY comma-separated tags."
            ),
            (
                "Tag this scene in the style of a Rule34 artist upload. "
                "Lead with participants (1girl, 1boy, etc.), then appearance/role tags, then the explicit act, "
                "then body details, position, props/objects mentioned, and setting. "
                "Include ALL context from the description (occupation, appearance, devices, location). "
                "Be graphic and specific. Output ONLY comma-separated Danbooru tags."
            ),
            (
                "Create a detailed explicit tag list for this description as if tagging it on Danbooru. "
                "Prioritize: participant tags, role/occupation tags, appearance tags, act tags "
                "(fellatio, sex, rape, gangbang, doggystyle...), anatomy tags, props/objects, "
                "expression tags (ahegao, moaning), setting, and clothing state. "
                "Include interracial, webcam, recording, streaming, or phone tags if the description mentions them. "
                "Output ONLY comma-separated tags, no explanations."
            ),
        ]
        extreme_focuses = [
            "Focus on architecture, lighting, and spatial composition. Prefer a different scene angle than the first pass.",
            "Focus on a broader cinematic reinterpretation. Add background structure, perspective, and props that still fit the description.",
            "Focus on an alternate scene reading. Keep the subject the same, but shift the emphasis toward environment and composition.",
        ]

        def pick_focus(options: list[str]) -> str:
            seed = f"{creativity}|{attempt_index}|{description.strip().lower()}".encode("utf-8")
            digest = hashlib.md5(seed).hexdigest()
            index = int(digest[:8], 16) % len(options)
            return options[index]

        if creativity == "creative":
            return (
                f"{description}\n\n"
                f"{pick_focus(creative_focuses)}"
            )

        if creativity == "mature":
            focus = pick_focus(mature_focuses)
            return (
                f"Description: {description}\n\n"
                f"{focus}"
            )

        focus = pick_focus(extreme_focuses)
        if attempt_index == 0:
            return (
                f"{description}\n\n"
                f"{focus}"
            )

        return (
            f"{description}\n\n"
            f"{focus} Make this pass deliberately different from the first one while staying grounded in the description."
        )

    def _enrich_tags_with_scene_ideas(
        self,
        description: str,
        tags: list[str],
        creativity: str,
        max_tags: int,
    ) -> list[str]:
        """Add vocabulary-safe creative scene tags using description-driven heuristics."""
        if creativity == "safe":
            return tags[:max_tags]

        if creativity == "mature":
            return self._enrich_mature_tags(description, tags, max_tags)

        seen = set(tags)
        enriched = list(tags)
        desc = description.lower()

        direct_trigger_map = [
            (
                ["nun", "church", "religious"],
                [
                    ["habit", "veil", "cross_necklace"],
                    ["church", "altar", "stained_glass"],
                    ["praying", "kneeling", "rosary"],
                    ["pew", "sanctuary", "candlelight"],
                ],
            ),
            (
                ["zombie", "ghoul", "undead", "monster"],
                [
                    "zombie",
                    "undead",
                    "monster",
                    "monster_girl",
                    "graveyard",
                    "grave",
                    "tombstone",
                ],
            ),
            (
                ["cemetery", "graveyard", "grave", "tomb"],
                ["graveyard", "grave", "tombstone", "cross", "outdoors", "ruins", "crow"],
            ),
            (
                ["fellatio", "oral", "sex", "lewd"],
                [
                    "open_mouth",
                    "oral",
                    "sex",
                    "1boy",
                    "2boys",
                    "multiple_boys",
                    "penis",
                    "erection",
                    "looking_at_penis",
                    "cum",
                    "cum_in_mouth",
                    "nude",
                    "spread_legs",
                ],
            ),
        ]

        cinematic_trigger_map = [
            (
                [
                    "scary",
                    "horror",
                    "creepy",
                    "ambience",
                    "atmosphere",
                    "dark ambience",
                    "dark atmosphere",
                ],
                [
                    "horror_(theme)",
                    "dark",
                    "darkness",
                    "shadow",
                    "moonlight",
                    "backlighting",
                    "candlelight",
                    "fog",
                    "smoke",
                    "embers",
                    "dutch_angle",
                    "wide_shot",
                    "depth_of_field",
                ],
            ),
        ]

        trigger_map = list(direct_trigger_map)
        if creativity in {"creative", "extreme"}:
            trigger_map.extend(cinematic_trigger_map)

        for triggers, candidates in trigger_map:
            if not any(trigger in desc for trigger in triggers):
                continue
            selected_candidates = candidates
            if candidates and isinstance(candidates[0], list):
                seed = f"{creativity}|{desc}".encode("utf-8")
                digest = hashlib.md5(seed).hexdigest()
                selected_candidates = candidates[int(digest[:8], 16) % len(candidates)]

            for candidate in selected_candidates:
                if candidate in seen:
                    continue
                if self.known_tags and candidate not in self.known_tags:
                    continue
                seen.add(candidate)
                enriched.append(candidate)
                if len(enriched) >= max_tags:
                    return enriched

        return enriched

    # Tags that belong exclusively to one position/act — used to strip conflicts.
    # Also includes associated oral/finish tags so they get cleaned from non-matching scenes.
    _POSITION_CONFLICT_GROUPS: list[list[str]] = [
        # Oral — these tags make no sense outside a fellatio/blowjob scene
        [
            "kneeling", "fellatio", "blowjob", "oral", "open_mouth",
            "penis_in_mouth", "deepthroat", "cum_in_mouth", "gagging",
            "tongue_out", "saliva", "drooling", "licking",
        ],
        # Doggystyle
        ["doggystyle", "sex_from_behind", "bent_over", "on_all_fours"],
        # Missionary
        ["missionary", "on_back", "legs_up"],
        # Cowgirl
        ["cowgirl_position", "reverse_cowgirl", "straddling", "riding"],
        # 69
        ["69", "sixty-nine"],
    ]

    @staticmethod
    def _detect_primary_act(desc: str, existing_tags: list[str]) -> str | None:
        """Return the name of the primary sex act detected in description or tags.

        Priority order matters: more specific acts must come before generic 'sex'.
        'on top of' / 'on top' / 'riding' → cowgirl before falling through to sex.
        """
        combined = desc + " " + " ".join(existing_tags)
        act_keywords: list[tuple[str, list[str]]] = [
            ("fellatio",    ["fellatio", "blowjob", "oral", "deepthroat", "suck", "bj"]),
            ("doggystyle",  ["doggystyle", "doggy", "from behind", "from_behind", "sex_from_behind"]),
            ("cowgirl",     ["cowgirl", "on top of", "on top", "riding", "ride", "straddling"]),
            ("missionary",  ["missionary"]),
            ("anal",        ["anal", "anal_sex"]),
            ("gangbang",    ["gangbang", "group_sex", "multiple men", "orgy"]),
            ("rape",        ["rape", "forced", "non-con", "non_con"]),
            ("69",          ["69", "sixty-nine"]),
            ("paizuri",     ["paizuri", "titfuck", "breast_sex"]),
            ("tentacle",    ["tentacle", "tentacles"]),
            ("sex",         ["sex", "intercourse", "fuck", "penetration", "vaginal"]),
        ]
        for act_name, keywords in act_keywords:
            if any(kw in combined for kw in keywords):
                return act_name
        return None

    def _strip_conflicting_positions(self, tags: list[str], primary_act: str | None) -> list[str]:
        """Remove position tags that contradict the primary act."""
        if primary_act is None:
            return tags

        # Which conflict group does the primary act belong to?
        primary_group: set[str] = set()
        for group in self._POSITION_CONFLICT_GROUPS:
            if any(kw in primary_act or primary_act in kw for kw in group):
                primary_group = set(group)
                break
        # Also seed from the actual tags present
        for group in self._POSITION_CONFLICT_GROUPS:
            if any(t in group for t in tags):
                if any(kw in primary_act or primary_act in kw for kw in group):
                    primary_group = set(group)
                    break

        if not primary_group:
            return tags

        # Collect all tags NOT in a conflicting group
        other_groups: set[str] = set()
        for group in self._POSITION_CONFLICT_GROUPS:
            if set(group) != primary_group:
                other_groups.update(group)

        return [t for t in tags if t not in other_groups]

    def _enrich_mature_tags(self, description: str, tags: list[str], max_tags: int) -> list[str]:
        """Add explicit Danbooru/Rule34 tags based on description cues.

        Strategy:
        1. Detect the ONE primary act.
        2. Inject participant count tags.
        3. Fire ONLY the enrichment pack that matches the primary act (plus setting/non-act extras).
        4. Strip any positions that contradict the primary act.
        """
        desc = description.lower()

        def _allowed(tag: str) -> bool:
            if self.danbooru_tags:
                return tag in self.danbooru_tags
            return True

        # --- Step 1: detect primary act ---
        primary_act = self._detect_primary_act(desc, tags)

        # --- Step 2: strip conflicting positions already in LLM output ---
        tags = self._strip_conflicting_positions(tags, primary_act)

        seen = set(tags)
        enriched = list(tags)

        # --- Step 3: inject participant count tags at front ---
        participant_tags = self._extract_actor_tags_from_description(description)
        for pt in participant_tags:
            if pt not in seen and _allowed(pt):
                seen.add(pt)
                enriched.insert(0, pt)

        def _add_pack(pack: list[str]) -> bool:
            """Add tags from pack; return True if max_tags reached."""
            for candidate in pack:
                if candidate in seen or not _allowed(candidate):
                    continue
                seen.add(candidate)
                enriched.append(candidate)
                if len(enriched) >= max_tags:
                    return True
            return False

        pack_seed = int(hashlib.md5(f"mature|{desc}".encode("utf-8")).hexdigest()[:8], 16)

        # --- Step 4: primary-act enrichment ---
        act_packs: dict[str, list[list[str]]] = {
            "fellatio": [
                ["fellatio", "oral", "penis", "erection", "open_mouth", "saliva", "eye_contact", "blush", "kneeling", "moaning", "nude"],
                ["fellatio", "deepthroat", "saliva", "gagging", "tears", "ahegao", "nude", "eye_contact"],
                ["fellatio", "licking", "tongue_out", "wet", "drooling", "blush", "panting", "eye_contact", "kneeling"],
            ],
            "doggystyle": [
                ["doggystyle", "sex_from_behind", "ass", "penis", "penetration", "moaning", "bent_over", "panting", "nude"],
                ["doggystyle", "on_all_fours", "ass", "sex_from_behind", "cum", "creampie", "moaning", "blush"],
            ],
            "missionary": [
                ["missionary", "on_back", "legs_up", "spread_legs", "penis", "penetration", "moaning", "blush", "nude"],
                ["missionary", "on_back", "legs_up", "cum", "creampie", "ahegao", "blush", "eye_contact"],
            ],
            "cowgirl": [
                ["cowgirl_position", "straddling", "riding", "penis", "breasts", "moaning", "blush", "sweat", "nude"],
                ["reverse_cowgirl", "straddling", "riding", "ass", "penis", "moaning", "nude", "blush"],
            ],
            "anal": [
                ["anal", "ass", "penis", "spread_ass", "moaning", "blush", "bent_over", "nude"],
                ["anal", "doggystyle", "ass", "penetration", "nude", "panting", "cum", "creampie"],
            ],
            "gangbang": [
                ["gangbang", "multiple_boys", "group_sex", "cum", "cum_on_body", "double_penetration", "ahegao", "nude"],
                ["gangbang", "group_sex", "multiple_boys", "rape", "spread_legs", "cum_on_face", "moaning"],
            ],
            "rape": [
                ["rape", "forced", "tears", "crying", "restrained", "spread_legs", "nude", "moaning"],
                ["rape", "restraints", "bondage", "submission", "dominance", "power_dynamic", "moaning", "nude"],
            ],
            "69": [
                ["69", "oral", "cunnilingus", "fellatio", "lying", "nude", "moaning", "blush"],
            ],
            "paizuri": [
                ["paizuri", "large_breasts", "penis", "erection", "breast_press", "moaning", "nude"],
            ],
            "tentacle": [
                ["tentacles", "tentacle_sex", "rape", "nude", "spread_legs", "restrained", "ahegao", "moaning"],
                ["tentacles", "penetration", "multiple_penetrations", "cum", "moaning", "nude"],
            ],
            "sex": [
                ["sex", "vaginal", "penis", "spread_legs", "moaning", "nude", "blush", "eye_contact"],
                ["sex", "penetration", "moaning", "blush", "nude", "cum", "creampie"],
            ],
        }

        if primary_act and primary_act in act_packs:
            packs = act_packs[primary_act]
            pack = packs[pack_seed % len(packs)]
            if _add_pack(pack):
                return enriched

        # --- Step 5: streamer / streaming / live context ---
        if any(w in desc for w in ["streamer", "streaming", "online", "live", "camgirl", "webcam", "onlyfans"]):
            stream_extras = [
                ["streamer", "webcam", "computer", "indoors", "recording"],
                ["live_stream", "desk", "microphone", "webcam", "recording"],
                ["webcam", "phone", "recording", "camera", "screen", "monitor"],
            ]
            if _add_pack(stream_extras[pack_seed % len(stream_extras)]):
                return enriched

        # --- Step 5b: headphones — only when explicitly mentioned ---
        if any(w in desc for w in ["headphones", "headset", "headphone", "gamer", "gaming"]):
            if _add_pack(["headphones"]):
                return enriched

        # --- Step 6: ugly / ugly_man / appearance context ---
        if any(w in desc for w in ["ugly", "ugly_man", "ugly bastard", "uggo", "hideous", "gross"]):
            ugly_extras = [
                ["ugly_man", "old_man", "wrinkles", "unshaven"],
                ["ugly_man", "chubby", "sweat", "hairy"],
                ["ugly_man", "bald", "scars", "fat"],
            ]
            if _add_pack(ugly_extras[pack_seed % len(ugly_extras)]):
                return enriched

        # --- Step 7: interracial ---
        if any(w in desc for w in ["interracial", "blacked", "bbc", "dark skin", "dark-skinned", "african", "ebony"]):
            interracial_extras = [
                ["interracial", "dark_skin", "black_hair", "dark-skinned_male", "muscular"],
                ["interracial", "dark_skin", "dark-skinned_female", "curvy"],
                ["interracial", "dark_skin", "race_play", "size_difference"],
            ]
            if _add_pack(interracial_extras[pack_seed % len(interracial_extras)]):
                return enriched

        # --- Step 8: phone / recording / camera context ---
        if any(w in desc for w in ["phone", "smartphone", "recording", "camera", "recorded", "filming", "video", "cellphone"]):
            phone_extras = [
                ["phone", "holding_phone", "smartphone", "recording", "camera"],
                ["phone", "webcam", "screen", "video", "selfie"],
                ["phone", "camera", "filming", "pov", "cameraman"],
            ]
            if _add_pack(phone_extras[pack_seed % len(phone_extras)]):
                return enriched

        # --- Step 9: nun/religious context ---
        if any(w in desc for w in ["nun", "priest", "church", "religious", "habit", "convent"]):
            nun_extras = [
                ["nun", "habit", "white_habit", "cross", "submission", "power_dynamic", "indoors", "church"],
                ["nun", "religious_clothing", "veil", "forbidden", "indoors"],
            ]
            if _add_pack(nun_extras[pack_seed % len(nun_extras)]):
                return enriched

        # --- Step 10: cum/finish tags ---
        if any(w in desc for w in ["cum", "cumshot", "creampie", "facial", "finish", "ejaculate"]):
            cum_extras = [
                ["cum", "cumshot", "cum_on_face", "cum_on_body", "ahegao", "open_mouth", "blush"],
                ["creampie", "cum_inside", "moaning", "blush"],
            ]
            if _add_pack(cum_extras[pack_seed % len(cum_extras)]):
                return enriched

        # --- Step 11: futanari ---
        if any(w in desc for w in ["futanari", "futa", "dickgirl", "femboy", "trap"]):
            futa_extras = [["futanari", "penis", "breasts", "erection", "nude"]]
            if _add_pack(futa_extras[0]):
                return enriched

        # --- Step 12: goth / emo / alternative style ---
        if any(w in desc for w in ["goth", "gothic", "emo", "punk", "alternative", "alt girl", "dark style"]):
            goth_extras = [
                ["gothic", "black_hair", "black_clothing", "fishnet", "choker", "dark_makeup", "pale_skin"],
                ["emo", "black_nails", "eyeliner", "torn_stockings", "platform_boots", "dark_lipstick"],
                ["goth", "lace", "corset", "thighhighs", "choker", "fishnet_stockings", "black_dress"],
            ]
            if _add_pack(goth_extras[pack_seed % len(goth_extras)]):
                return enriched

        # --- Step 13: setting/atmosphere extras ---
        if any(w in desc for w in ["bedroom", "bed", "hotel", "room"]):
            if _add_pack(["bed", "indoors", "night", "soft_lighting"]):
                return enriched

        return enriched

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
            if self.known_tags and tag not in self.known_tags:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _extract_actor_tags_from_description(self, description: str) -> list[str]:
        """Infer participant tags so mature mode keeps visible actors in the output."""
        desc = description.lower()
        tags: list[str] = []

        def add(tag: str) -> None:
            if self.known_tags and tag not in self.known_tags:
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

        if "1 on 1" in desc or "one on one" in desc:
            if not tags:
                add("1boy")
                add("1girl")
            elif len(tags) == 1:
                add("1girl" if tags[0] == "1boy" else "1boy")

        return tags
    
    def _parse_tags(self, raw_response: str, creativity: str = DEFAULT_CREATIVITY) -> list[str]:
        """Extract tags from raw LLM response.

        Filtering priority (first match wins):
        1. If the full Danbooru CSV whitelist is loaded → use it for ALL modes.
           (The CSV contains the real Danbooru vocabulary including explicit tags.)
        2. Else if known_tags (WD-tagger SFW vocab) is loaded AND mode is not NSFW
           → fall back to that narrow filter.
        3. Otherwise → accept all tags (no filter).

        The WD-tagger SFW filter is never used for NSFW modes because it silently
        drops almost every explicit tag.
        """
        raw_response = self._clean_response_text(raw_response)

        is_nsfw = creativity in self._NSFW_MODES
        if self.danbooru_tags:
            # Full Danbooru CSV available — use for all modes
            use_danbooru = True
            use_known    = False
        elif self.known_tags and not is_nsfw:
            # Only narrow SFW vocab, and we're not in an NSFW mode
            use_danbooru = False
            use_known    = True
        else:
            use_danbooru = False
            use_known    = False

        tags: list[str] = []
        seen: set[str] = set()

        for chunk in re.split(r"[\n,\.]+", raw_response):
            line = chunk.strip()
            if not line:
                continue
            # Remove numbered list markers like "1. ", "2) " but preserve tag numbers like "1girl"
            line = re.sub(r"^[0-9]+[.)]\s*|^[\-\*]\s+", "", line)
            line = line.strip().strip(".")
            if not line or any(c in line for c in [":", "=", ">"]):
                continue
            normalized = line.lower().replace(" ", "_")
            if normalized in seen:
                continue

            if use_danbooru and normalized not in self.danbooru_tags:
                continue
            if use_known and normalized not in self.known_tags:
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
