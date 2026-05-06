from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore

from backend.description_tagger import DescriptionTagResult, get_description_tagger
from backend.tag_utils import IMAGE_EXTENSIONS


class DescriptionTagWorker(QtCore.QThread):
    finished = QtCore.Signal(DescriptionTagResult)
    error = QtCore.Signal(str)

    def __init__(self, description: str, model: str, creativity: str) -> None:
        super().__init__()
        self.description = description
        self.model = model
        self.creativity = creativity

    def run(self) -> None:
        try:
            tagger = get_description_tagger(model=self.model)
            result = tagger.generate_tags(self.description, creativity=self.creativity)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ImageLoadWorker(QtCore.QThread):
    finished = QtCore.Signal(list, list)
    progress = QtCore.Signal(int)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths

    def run(self) -> None:
        valid: list[Path] = []
        skipped: list[str] = []
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped.append(path.name)
                self.progress.emit(int((i + 1) / total * 100))
                continue
            try:
                Image.open(path).convert("RGB")
                valid.append(path)
            except (UnidentifiedImageError, OSError):
                skipped.append(path.name)
            self.progress.emit(int((i + 1) / total * 100))
        self.finished.emit(valid, skipped)
