from __future__ import annotations

import csv
import inspect
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image


MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    confidence: float
    category: int


@dataclass(frozen=True)
class TagRecord:
    name: str
    category: int


def _cache_dir() -> Path:
    return Path.home() / ".img_tagger"


def _download_model_file(filename: str) -> Path:
    cache_root = _cache_dir() / MODEL_REPO.replace("/", "__")
    cache_root.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            cache_dir=str(cache_root),
        )
    )


def _load_tags(csv_path: Path) -> list[TagRecord]:
    records: list[TagRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        peek = handle.readline()
        handle.seek(0)
        if "name" in peek.lower() and "category" in peek.lower():
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    category = int((row.get("category") or "0").strip())
                except ValueError:
                    category = 0
                records.append(TagRecord(name=name, category=category))
            return records

        reader = csv.reader(handle)
        rows = list(reader)
        for row in rows:
            if not row:
                continue
            if len(row) >= 3 and row[0].lower() == "name":
                continue
            name = row[0].strip()
            if not name:
                continue
            try:
                category = int(row[1]) if len(row) > 1 else 0
            except ValueError:
                category = 0
            records.append(TagRecord(name=name, category=category))
    return records


def mcut_threshold(probs: np.ndarray) -> float:
    if probs.size < 2:
        return float(probs.max()) if probs.size else 0.0
    sorted_probs = np.sort(probs)[::-1]
    diffs = sorted_probs[:-1] - sorted_probs[1:]
    cut_index = int(np.argmax(diffs))
    return float((sorted_probs[cut_index] + sorted_probs[cut_index + 1]) / 2.0)


def _make_square(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - resized.size[0]) // 2, (size - resized.size[1]) // 2)
    square.paste(resized, offset)
    return square


class AnimeTagger:
    def __init__(self) -> None:
        model_path = _download_model_file(MODEL_FILE)
        tags_path = _download_model_file(TAGS_FILE)
        self.tags = _load_tags(tags_path)
        self.session = ort.InferenceSession(str(model_path), providers=self._providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    @staticmethod
    def _providers() -> list[str]:
        available = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _target_size(self) -> int:
        shape = self.input_shape
        candidates = [shape[-2], shape[-1]] if len(shape) >= 4 else []
        for candidate in candidates:
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return 448

    def _prepare_image(self, image: Image.Image, normalize_pixels: bool) -> np.ndarray:
        size = self._target_size()
        square = _make_square(image.convert("RGB"), size)
        array = np.asarray(square, dtype=np.float32)
        array = array[:, :, ::-1]
        if normalize_pixels:
            array = array / 255.0
        array = np.expand_dims(array, axis=0)
        return array

    def predict(
        self,
        image: Image.Image,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        normalize_pixels: bool = False,
        use_mcut: bool = False,
        limit: int | None = None,
    ) -> list[TagPrediction]:
        inputs = self._prepare_image(image, normalize_pixels=normalize_pixels)
        raw = self.session.run([self.output_name], {self.input_name: inputs})[0]
        scores = 1.0 / (1.0 + np.exp(-raw[0]))

        predictions: list[TagPrediction] = []
        general_candidates: list[TagPrediction] = []
        character_candidates: list[TagPrediction] = []

        for index, score in enumerate(scores[: len(self.tags)]):
            record = self.tags[index]
            tag_name = record.name.replace("_", " ")
            prediction = TagPrediction(tag=tag_name, confidence=float(score), category=record.category)
            if record.category == 9:
                continue
            if record.category == 4:
                character_candidates.append(prediction)
                continue
            general_candidates.append(prediction)

        if use_mcut:
            general_values = np.array([prediction.confidence for prediction in general_candidates], dtype=np.float32)
            if general_values.size:
                general_threshold = mcut_threshold(general_values)
            character_values = np.array([prediction.confidence for prediction in character_candidates], dtype=np.float32)
            if character_values.size:
                character_threshold = max(0.15, mcut_threshold(character_values))

        for prediction in general_candidates:
            if prediction.confidence >= general_threshold:
                predictions.append(prediction)

        for prediction in character_candidates:
            if prediction.confidence >= character_threshold:
                predictions.append(prediction)

        predictions.sort(key=lambda item: item.confidence, reverse=True)
        if limit is not None:
            predictions = predictions[:limit]
        return predictions


def predict_tags(
    tagger: AnimeTagger,
    image: Image.Image,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    normalize_pixels: bool = False,
    use_mcut: bool = False,
    limit: int | None = None,
) -> list[TagPrediction]:
    parameters = inspect.signature(tagger.predict).parameters
    kwargs: dict[str, object] = {}

    if "general_threshold" in parameters:
        kwargs["general_threshold"] = general_threshold
    elif "threshold" in parameters:
        kwargs["threshold"] = general_threshold

    if "character_threshold" in parameters:
        kwargs["character_threshold"] = character_threshold

    if "normalize_pixels" in parameters:
        kwargs["normalize_pixels"] = normalize_pixels

    if "use_mcut" in parameters:
        kwargs["use_mcut"] = use_mcut

    if "limit" in parameters:
        kwargs["limit"] = limit

    return tagger.predict(image, **kwargs)


@lru_cache(maxsize=1)
def get_tagger() -> AnimeTagger:
    return AnimeTagger()


def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def category_label(category: int) -> str:
    return {
        0: "general",
        4: "character",
        9: "rating",
    }.get(category, f"category_{category}")


def caption_from_predictions(
    predictions: Iterable[TagPrediction],
    blacklist: Sequence[str] | None = None,
    whitelist: Sequence[str] | None = None,
    include_scores: bool = False,
) -> str:
    blacklist_set = {tag.strip().lower() for tag in (blacklist or []) if tag.strip()}
    whitelist_set = {tag.strip().lower() for tag in (whitelist or []) if tag.strip()}
    parts: list[str] = []
    for prediction in predictions:
        normalized = prediction.tag.lower().strip()
        if blacklist_set and normalized in blacklist_set:
            continue
        if whitelist_set and normalized not in whitelist_set:
            continue
        if include_scores:
            parts.append(f"{prediction.tag}:{prediction.confidence:.3f}")
        else:
            parts.append(prediction.tag)
    return ", ".join(parts)
