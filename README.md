# Local Anime Tagger

Local anime image captioning tool inspired by WD14 / OneTrainer workflows.

## What it does

- Drops in one or more images.
- Returns Danbooru-style tags with confidence scores.
- Lets you edit, sort, filter, and copy tags before export.
- Exports one `.txt` caption file per image.
- Supports batch processing from a local folder.
- Runs locally on Windows and uses GPU automatically if `onnxruntime-gpu` is installed and CUDA is available.

## Recommended stack

- Backend: Python + FastAPI + ONNX Runtime.
- Frontend: Streamlit.
- Model: `SmilingWolf/wd-swinv2-tagger-v3` from Hugging Face.

This repo includes a minimal working version that keeps the inference logic in a shared local service module. The FastAPI backend is available as an optional local API if you want to integrate another UI later.

## Project layout

- `backend/` - tagger service and optional local API.
- `frontend/` - Streamlit editor, preview, and export UI.
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

## Notes

- The model downloads on first run and is cached locally.
- The UI includes a caption editor, category filtering, blacklist/whitelist fields, batch folder processing, category-specific thresholds, optional MCut thresholding, and export to a zip or a local output folder.
- If captions look off, try toggling the pixel normalization option in the sidebar. Some ONNX tagger exports behave better with raw 0-255 pixels and others with 0-1 normalized pixels.
