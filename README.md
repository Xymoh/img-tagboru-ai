# Local Anime Tagger

Local anime image captioning tool inspired by WD14 / OneTrainer workflows.

## What it does

- Drops in one or more images.
- Returns Danbooru-style tags with confidence scores.
- Lets you edit, sort, filter, and copy tags before export.
- Exports one `.txt` caption file per image.
- Supports batch processing from a local folder.
- Runs locally on Windows and uses GPU automatically if `onnxruntime-gpu` is installed and CUDA is available.
- **NEW:** Generate Danbooru tags from text descriptions using local LLM (Ollama) — no restricted keywords, fully unrestricted.

## Features

### Image-to-Tags
- Load images via file picker, folder, drag-and-drop, or paste from clipboard
- ONNX-based WD-SwinV2 tagger with category-specific confidence thresholds
- MCut thresholding for automatic cutoff detection
- Tag filtering (blacklist/whitelist)
- Batch processing with progress tracking

### Description-to-Tags
- Describe a scene in English — AI generates Danbooru-style tags from your description
- 3 creativity modes: Safe (SFW), Creative (mild NSFW), Mature (explicit NSFW)
- No API calls, fully offline
- Vocabulary-grounded prompting backed by `danbooru_tags_post_count.csv` (99,995 tags)
- Multi-step post-processing: whitelist validation, relevance gate, concept/act expansions, semantic dedup, pose-conflict resolution
- Uses Ollama + an abliterated Qwen3 model running locally

## PC Requirements for Description-to-Tags

The description-to-tags feature requires a local LLM running via **Ollama**:

**Minimum Hardware:**
- **RAM:** 16 GB (32 GB recommended)
- **CPU:** Modern multi-core processor
- **GPU:** NVIDIA/AMD with 16 GB VRAM recommended (8 GB workable with smaller models)
- **Disk:** ~10 GB per model

**Setup:**
1. Download and install **Ollama** from [ollama.ai](https://ollama.ai)
2. Start Ollama: `ollama serve`
3. Pull the recommended model: `ollama pull richardyoung/qwen3-14b-abliterated`
4. First inference takes 30s-2min (model load); subsequent calls average ~4s per run

The Description Tagger applies a deterministic post-processing pipeline (relevance gate, concept/act expansions, pose-conflict resolver) on top of the LLM output, so even with a small or occasionally-refusing model the results stay coherent. See the in-app Help dialog for model alternatives, test results, and tradeoffs.

**Note:** This feature is optional. Image-to-tags works without Ollama.

## Recommended stack

- Backend: Python + FastAPI + ONNX Runtime.
- Frontend: PySide6 desktop app.
- Model: `SmilingWolf/wd-swinv2-tagger-v3` from Hugging Face.

This repo is a standalone desktop application. The FastAPI backend (`backend/api.py`) is also available as an optional local API if you want to integrate a custom UI.

## Project layout

- `backend/` — tagger service, description tagger (LLM), tag index, and optional FastAPI.
- `frontend/native/` — PySide6 desktop app (main entry point: `main_window.py`).
- `scripts/` — batch test utilities.
- `requirements.txt` — Python dependencies.

## Setup on Windows

1. Install Python 3.10+.
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. If you want GPU support, replace CPU runtime with the GPU build:

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

6. Start the UI:

```powershell
python frontend/native/main_window.py
```

Optional: start the API server in another terminal.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

If `uvicorn` is not recognized, use the venv Python module form above instead of the bare command.

## Build a Windows exe

This project can be packaged as a standalone Windows desktop exe.

1. Install the dependencies in your virtual environment.
2. Run the build script:

```powershell
.\build_exe.ps1
```

3. The packaged app will be written to `dist\Img-Tagboru.exe`.

GitHub Releases are also supported: push a tag like `v1.0.0` and the Windows build workflow will publish `dist\Img-Tagboru.exe` as a release asset.

Notes:

- The exe launches the native PySide6 desktop app.
- The first run still downloads the model files to the user's cache folder.
- If Windows Defender or SmartScreen warns about the exe, that is normal for unsigned local builds.

## FastAPI Endpoints

Start the API server:

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

### `GET /health`

Returns a simple health-check response. Useful for monitoring and readiness probes.

**Response** `200 OK`
```json
{"status": "ok"}
```

### `POST /tag`

Tags a single image and returns Danbooru-style tags with confidence scores.

**Request** — `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | *required* | Image file (PNG, JPG, WebP, BMP, GIF) |
| `general_threshold` | float | `0.35` | Minimum confidence for general tags (0.0–1.0) |
| `character_threshold` | float | `0.85` | Minimum confidence for character tags (0.0–1.0) |
| `normalize_pixels` | bool | `false` | Normalize pixels to 0–1 range before inference |
| `use_mcut` | bool | `false` | Use MCut automatic threshold detection |
| `limit` | int | `80` | Maximum number of tags to return (0 = unlimited) |

**Response** `200 OK`
```json
{
  "caption": "1girl, smile, blue_eyes, solo, ...",
  "tags": [
    {"tag": "1girl", "confidence": 0.9876, "category": 0, "category_label": "general"},
    {"tag": "hatsune_miku", "confidence": 0.9521, "category": 4, "category_label": "character"}
  ]
}
```

**Category labels:** `general` (0), `artist` (1), `copyright` (3), `character` (4), `meta` (5).

### CORS

The API enables CORS for `http://localhost:5173` and `http://127.0.0.1:5173` (Vite dev server defaults). Configure additional origins in [`backend/api.py`](backend/api.py:19).

### Connect to a Vite app

Example fetch from a Vite frontend:

```ts
const formData = new FormData();
formData.append('file', file);
formData.append('general_threshold', '0.35');

const response = await fetch('http://127.0.0.1:8000/tag', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
// result.caption → "1girl, smile, blue_eyes, ..."
// result.tags → [{tag, confidence, category, category_label}, ...]
```
## Notes

- The model downloads on first run and is cached locally.
- The UI includes a caption editor, category filtering, blacklist/whitelist fields, batch folder processing, category-specific thresholds, optional MCut thresholding, and export to a zip or a local output folder.
- If captions look off, try toggling the pixel normalization option in the sidebar. Some ONNX tagger exports behave better with raw 0-255 pixels and others with 0-1 normalized pixels.
