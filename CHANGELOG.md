# Changelog

All notable changes to Img-Tagboru are documented in this file.

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
