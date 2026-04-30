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

### Description-to-Tags (NEW)
- Describe what you want to see on an image
- AI generates corresponding Danbooru tags using a local LLM
- No API calls, fully offline
- Unrestricted tag generation (includes NSFW keywords when relevant)
- Uses Ollama + Qwen (or Llama) running locally

## PC Requirements for Description-to-Tags

The description-to-tags feature requires a local LLM running via **Ollama**:

**Minimum Hardware:**
- **RAM:** 8 GB (16 GB recommended for better performance)
- **CPU:** Modern multi-core processor
- **GPU:** Optional but strongly recommended (NVIDIA/AMD/Intel with proper drivers)
- **Disk:** 10-20 GB free for model storage

**Setup:**
1. Download and install **Ollama** from [ollama.ai](https://ollama.ai)
2. Start Ollama: `ollama serve`
3. In another terminal, pull a model: `ollama pull qwen2:7b` (recommended) or `ollama pull llama2`
4. First inference takes 30s-2min depending on hardware; subsequent calls are faster

**Note:** This feature is optional. Image-to-tags works without Ollama.

## Recommended stack

- Backend: Python + FastAPI + ONNX Runtime.
- Frontend: PySide6.
- Model: `SmilingWolf/wd-swinv2-tagger-v3` from Hugging Face.

This repo includes a standalone desktop app that keeps the inference logic in a shared local service module. The FastAPI backend is available as an optional local API if you want to integrate another UI later.

## Project layout

- `backend/` - tagger service and optional local API.
- `frontend/` - PySide6 editor, preview, and export UI.
- `requirements.txt` - Python dependencies.

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
python frontend/native_app.py
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

3. The packaged app will be written to `dist\img-tagger.exe`.

GitHub Releases are also supported: push a tag like `v1.0.0` and the Windows build workflow will publish `dist\img-tagger.exe` as a release asset.

Notes:

- The exe launches the native PySide6 desktop app.
- The first run still downloads the model files to the user's cache folder.
- If Windows Defender or SmartScreen warns about the exe, that is normal for unsigned local builds.

## Connect to a Vite app

Your Vite frontend can talk to this app over the local API. Use `http://127.0.0.1:8000/tag` for single-image tagging and `http://127.0.0.1:8000/health` for a quick availability check.

Because the API enables CORS for `http://localhost:5173` and `http://127.0.0.1:5173`, a local Vite dev server can fetch captions directly.

Example fetch from Vite:

```ts
const formData = new FormData();
formData.append('file', file);
formData.append('threshold', '0.35');

const response = await fetch('http://127.0.0.1:8000/tag', {
	method: 'POST',
	body: formData,
});

const result = await response.json();
```
## Notes

- The model downloads on first run and is cached locally.
- The UI includes a caption editor, category filtering, blacklist/whitelist fields, batch folder processing, category-specific thresholds, optional MCut thresholding, and export to a zip or a local output folder.
- If captions look off, try toggling the pixel normalization option in the sidebar. Some ONNX tagger exports behave better with raw 0-255 pixels and others with 0-1 normalized pixels.
