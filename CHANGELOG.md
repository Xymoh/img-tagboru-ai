# Changelog

All notable changes to Img-Tagboru are documented in this file.

## [Unreleased]

### Added
- **Relevance gate** in Description Tagger post-processing. Every tag is scored 0-3 against the user's description; score-0 tags (likely hallucinations) are dropped. Franchise tags like `magic_knight_rayearth`, `library_of_ruina`, `master_sword` no longer leak into unrelated prompts.
- **Concept expansion map** (`_CONCEPT_EXPANSIONS`) — 30+ archetype keywords (nun, witch, knight, maid, catgirl, orc, succubus, kitchen, cathedral, library, ...) map to plausible Danbooru tags, letting the LLM use e.g. `veil` for a nun description without that word being named.
- **NSFW act expansion map** (`_ACT_EXPANSIONS`) — 57 sex-act keywords expand to anatomy/position/reaction tags (e.g. `blowjob` → `fellatio, penis, open_mouth, saliva, kneeling, tongue_out`). Applied only in creative/mature modes; safe mode never sees anatomy.
- **Mutual-exclusion conflict resolver** — keeps only the highest-post_count tag within pose/viewpoint/framing/mouth/time-of-day groups (no more `standing + kneeling + lying` collisions).
- **Repetition-loop detector** in `_parse_tags` — truncates the raw response when the model stutters into `saliva, saliva, saliva, ...` loops.
- **Disambiguator auto-block** — tags shaped `tag_(context)` are dropped unless the parenthetical matches a word in the description.
- **Concept-expansion rescue path** — when the LLM returns nothing usable, synthesize tags from archetype and act expansions so the output is never empty.
- **Per-mode retry acceptance thresholds** — safe accepts ≥3 tags, creative ≥5, mature ≥6. Up to 3 attempts with +0.15 temperature per retry. Best result across attempts is kept.
- Expanded `_extract_literal_tags_from_description` from 15 phrases to 80+ (anatomy, body type, sex acts, participants, fluids, archetypes).

### Changed
- **Qwen3 `/no_think` directive** replaces the old `<think>` prefill. Generation time dropped from ~11s to ~4s per run (65% faster) because the reasoning block no longer consumes the `num_predict` budget.
- Prefill detection now scopes to qwen3 thinking variants only — qwen2.5, qwen3-Instruct-2507, and non-qwen models no longer receive the directive.
- Help dialog model recommendations updated with real tested numbers:
  - Qwen3-14B-Abliterated (default): 9.9 avg tags/run, 4s/run
  - Goonsai Qwen2.5-3B NSFW: 7.3 avg tags/run, 7s/run — listed as tested alternative for quick re-runs
  - huihui_ai/qwen3-abliterated:30b-a3b-q4_K_M marked as "not recommended" (thinking variant returns empty output)
  - Untested alternatives section with pull commands for the Instruct-2507 variant, qwen2.5-14b abliterated, and Cydonia 24B

### Fixed
- Franchise-name collisions were slipping through when tags shared a single token with the description. 2-token tags now require both tokens be supported; 3+ token tags require 75% of tokens supported.
- NSFW descriptions like "Orc forcing elf to give him blowjob" no longer return zero tags — act expansions let anatomy pass the relevance gate in creative/mature modes.

## [v1.3.1] — 2026-05-07

### Added
- **Model Manager dialog** — pull, list, and delete Ollama models from within the app (Description Tagger tab → ⚙️ Manage Models). No CLI needed.
- **Regex-based blacklist/whitelist** — wrap patterns in `/slashes/` (e.g. `/^bad_/`) in blacklist/whitelist fields for regex matching against tag names via `apply_filters()`.

### Fixed
- **Model Manager UI freeze** — pull and delete operations now run on a `ModelOperationWorker` background thread with an indeterminate progress bar, keeping the dialog responsive during long downloads.

### Changed
- Version bumped to `1.3.1` (window title, About dialog, `app.setApplicationVersion`).

## [v1.3.0] — 2026-05-07

### Added
- Danbooru-tag autocomplete in the caption editor backed by a prefix Trie (O(k) lookup independent of tag count).
- Background QThread CSV loading so the UI stays responsive while the 1M+ tag vocabulary loads.
- Watch-folder with `QFileSystemWatcher` + QTimer debouncing for automatic batch tagging of new images.
- Prompt Tools: Copy as Prompt, Build Negative Prompt, and Tag Frequency Dashboard.
- AI Metadata extraction from PNG info chunks (Stable Diffusion / NovelAI parameters).
- Visual toolbar groups with `QFrame.VLine` sunken dividers (5 logical groups).
- Undo/Redo stack (Ctrl+Z / Ctrl+Y) for tag table edits.
- Per-spinbox coloured SVG chevron icons.
- Export caption beside source file option.
- Drop overlay hint ("Drop images here") on drag enter.

### Changed
- Unified all version strings to `v1.3.0` (window title, About dialog, Qt metadata).
- Build artifact renamed from `img-tagger.exe` to `Img-Tagboru.exe`.
- Replaced all ad-hoc `print()` debug output with Python `logging` module (`logger.debug()`/`logger.info()`/`logger.warning()`).
- `requirements.txt`: fixed `uvicorn[standard]` → `uvicorn` (bracket extras syntax is `pip`-specific, not standard).

### Fixed
- Temp SVG accumulation: `styles.py` now writes SVG files once and registers `atexit` cleanup via `shutil.rmtree`.
- Web download security: added `Content-Length` validation (100 MiB cap) to `_download_web_image_to_temp`.
- README startup instructions: `frontend/native_app.py` → `frontend/native/main_window.py`.

### Removed
- Stale migration scripts: `scripts/add_method.py`, `scripts/fix_ui.py`.

## [v1.2.0] — 2025-12-01

### Added
- Description-to-Tags: generate Danbooru tags from text descriptions using local LLM (Ollama + Qwen / Llama).
- Mature-mode tag enrichment with sex-act detection, position deduplication, and actor extraction.
- `DescriptionTagger` class with creativity presets (`low`, `medium`, `high`, `mature`).
- `DescriptionTagWorker` QThread for non-blocking LLM inference.
- Help dialog with usage instructions.

### Changed
- Caption toolbar reorganised into workflow-ordered groups.
- SpinBox arrows replaced with custom SVG chevrons.

## [v1.1.0] — 2025-10-15

### Added
- Drag-and-drop from web browsers (HTML extraction of image candidates).
- Clipboard paste support (Ctrl+V).
- `QFileSystemWatcher` watch-folder groundwork.
- Zip export with in-memory `BytesIO`.

### Changed
- `TaggingResult` dataclass unified across desktop and Streamlit frontends.
- Category filtering via checkboxes instead of dropdown.

## [v1.0.0] — 2025-09-01

### Added
- Initial release.
- PySide6 desktop application with dark theme.
- ONNX-based WD-SwinV2 tagger with category-specific confidence thresholds.
- MCut thresholding for automatic cutoff detection.
- Tag table with inline include/exclude and manual rank editing.
- Batch folder processing with progress tracking.
- Blacklist/whitelist filtering.
- Export to individual `.txt` caption files.
- FastAPI backend (`/tag`, `/health`) with CORS for Vite integration.
- Streamlit web frontend (via `launcher.py`).
- `build_exe.ps1` PyInstaller packaging script.
- GitHub Actions Windows build workflow.
