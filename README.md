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

6. Start the UI:

```powershell
streamlit run frontend/app.py
```

Optional: start the API server in another terminal.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

If `uvicorn` is not recognized, use the venv Python module form above instead of the bare command.

## Build a Windows exe

This project can be packaged as a local Windows launcher exe that starts the Streamlit app on the user's machine.

1. Install the dependencies in your virtual environment.
2. Run the build script:

```powershell
.\build_exe.ps1
```

3. The packaged app will be written to `dist\img-tagger.exe`.

Notes:

- The exe is a local launcher for the Streamlit app, not a native desktop UI.
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
