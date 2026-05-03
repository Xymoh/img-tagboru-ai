from __future__ import annotations

from PySide6 import QtCore

from backend.description_tagger import DescriptionTagResult, get_description_tagger


class DescriptionTagWorker(QtCore.QThread):
    """Worker thread for description-to-tags generation (prevents UI freeze)."""

    finished = QtCore.Signal(DescriptionTagResult)
    error = QtCore.Signal(str)

    def __init__(self, description: str, model: str, creativity: str) -> None:
        super().__init__()
        self.description = description
        self.model = model
        self.creativity = creativity

    def run(self) -> None:
        """Run tag generation in background thread."""
        try:
            tagger = get_description_tagger(model=self.model)
            result = tagger.generate_tags(self.description, creativity=self.creativity)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
