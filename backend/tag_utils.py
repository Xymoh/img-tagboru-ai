from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image

from backend.tagger import TagPrediction, category_label


IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Standard DataFrame columns for tag tables
TAG_COLUMNS = ["include", "rank", "tag", "confidence", "category"]


@dataclass
class TaggingResult:
    """Unified result item used by both the desktop and Streamlit frontends."""

    name: str  # display filename
    path: Path | None  # filesystem path (None for in-memory uploads)
    image: Image.Image
    frame: pd.DataFrame
    caption: str

    @property
    def caption_filename(self) -> str:
        """Derived .txt filename for this result."""
        stem = Path(self.name).stem
        return f"{stem}.txt"


# ---------------------------------------------------------------------------
# Tag text / frame helpers
# ---------------------------------------------------------------------------


def split_tags(text: str) -> list[str]:
    """Split comma/newline-separated tag text into a clean list."""
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def frame_from_predictions(predictions: Sequence[TagPrediction]) -> pd.DataFrame:
    """Build a DataFrame from raw :class:`TagPrediction` objects."""
    rows: list[dict] = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "include": True,
                "rank": index + 1,
                "tag": prediction.tag,
                "confidence": round(prediction.confidence, 4),
                "category": category_label(prediction.category),
            }
        )
    if not rows:
        return pd.DataFrame(columns=TAG_COLUMNS)
    return pd.DataFrame(rows)


def apply_filters(frame: pd.DataFrame, blacklist: Iterable[str], whitelist: Iterable[str]) -> pd.DataFrame:
    """Mark tags for exclusion based on blacklist / whitelist.

    Returns a copy of *frame* with the ``include`` column adjusted.
    """
    blacklist_set = {tag.strip().lower() for tag in blacklist if tag.strip()}
    whitelist_set = {tag.strip().lower() for tag in whitelist if tag.strip()}

    def keep_row(tag: str) -> bool:
        normalized = str(tag).strip().lower()
        if blacklist_set and normalized in blacklist_set:
            return False
        if whitelist_set and normalized not in whitelist_set:
            return False
        return True

    result = frame.copy()
    result["include"] = result["include"] & result["tag"].map(keep_row)
    return result


def sort_frame(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    """Sort a tag DataFrame *in place copy* by the given mode.

    *sort_mode* may be ``"alphabetical"``, ``"confidence"``, or anything else
    (defaults to manual rank ordering).
    """
    if frame.empty:
        return frame.copy()
    if sort_mode == "alphabetical":
        return frame.sort_values(by=["tag", "confidence"], ascending=[True, False]).reset_index(drop=True)
    if sort_mode == "confidence":
        return frame.sort_values(by=["confidence", "tag"], ascending=[False, True]).reset_index(drop=True)
    return frame.sort_values(by=["rank", "confidence"], ascending=[True, False]).reset_index(drop=True)


def frame_to_caption(frame: pd.DataFrame, include_scores: bool = False) -> str:
    """Convert a tag DataFrame to a comma-separated caption string."""
    included = frame[frame["include"]].copy()
    if included.empty:
        return ""
    included = included.sort_values(by=["rank", "confidence"], ascending=[True, False])
    if include_scores:
        return ", ".join(f"{row.tag}:{row.confidence:.3f}" for row in included.itertuples())
    return ", ".join(included["tag"].tolist())


def caption_from_frame(frame: pd.DataFrame, include_scores: bool = False) -> str:
    """Alias for :func:`frame_to_caption` — kept for Streamlit compatibility."""
    return frame_to_caption(frame, include_scores=include_scores)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_zip_from_results(results: Iterable[TaggingResult]) -> bytes:
    """Create a ZIP of ``caption_filename → caption`` for every result."""
    return export_zip((r.caption_filename, r.caption) for r in results)


def export_zip(caption_pairs: Iterable[tuple[str, str]]) -> bytes:
    """Create a ZIP archive of ``(filename, content)`` caption pairs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in caption_pairs:
            archive.writestr(filename, content.encode("utf-8"))
    return buffer.getvalue()
