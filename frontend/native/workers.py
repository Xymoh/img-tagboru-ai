from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore

from backend.description_tagger import DescriptionTagResult, get_description_tagger
from backend.tag_utils import IMAGE_EXTENSIONS


class ModelOperationWorker(QtCore.QThread):
    """Pull or delete an Ollama model in a background thread to keep the UI responsive."""

    finished = QtCore.Signal(bool, str)
    """Emitted with (success, message) when the operation completes."""

    def __init__(self, operation: str, model_name: str) -> None:
        super().__init__()
        self._operation = operation  # "pull" or "delete"
        self._model_name = model_name

    def run(self) -> None:
        try:
            tagger = get_description_tagger()
            if self._operation == "pull":
                tagger.pull_model(self._model_name)
                self.finished.emit(True, f"Model '{self._model_name}' pulled successfully.")
            elif self._operation == "delete":
                tagger.delete_model(self._model_name)
                self.finished.emit(True, f"Model '{self._model_name}' deleted.")
            else:
                self.finished.emit(False, f"Unknown operation: {self._operation}")
        except Exception as e:
            self.finished.emit(False, str(e))


class DescriptionTagWorker(QtCore.QThread):
    finished = QtCore.Signal(DescriptionTagResult)
    error = QtCore.Signal(str)

    def __init__(
        self,
        description: str,
        model: str,
        creativity: str,
        post_count_threshold: int = 500,
        enrich_mode: bool = False,
    ) -> None:
        super().__init__()
        self.description = description
        self.model = model
        self.creativity = creativity
        self.post_count_threshold = post_count_threshold
        self.enrich_mode = enrich_mode

    def run(self) -> None:
        try:
            tagger = get_description_tagger(model=self.model)
            tagger.set_post_count_threshold(self.post_count_threshold)
            if self.enrich_mode:
                result = tagger.enrich_tags(self.description, creativity=self.creativity)
            else:
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
