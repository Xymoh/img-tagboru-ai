from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tagger import caption_from_predictions, category_label, get_tagger, image_from_bytes, predict_tags


app = FastAPI(title="Local Anime Tagger", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TagResponse(BaseModel):
    caption: str
    tags: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tag", response_model=TagResponse)
async def tag_image(
    file: UploadFile = File(...),
    general_threshold: float = Form(0.35),
    character_threshold: float = Form(0.85),
    normalize_pixels: bool = Form(False),
    use_mcut: bool = Form(False),
    limit: int = Form(80),
) -> TagResponse:
    data = await file.read()
    image = image_from_bytes(data)
    tagger = get_tagger()
    predictions = predict_tags(
        tagger,
        image,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        normalize_pixels=normalize_pixels,
        use_mcut=use_mcut,
        limit=limit if limit > 0 else None,
    )
    return TagResponse(
        caption=caption_from_predictions(predictions),
        tags=[
            {
                "tag": prediction.tag,
                "confidence": prediction.confidence,
                "category": prediction.category,
                "category_label": category_label(prediction.category),
            }
            for prediction in predictions
        ],
    )
