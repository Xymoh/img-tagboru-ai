from __future__ import annotations

import io
import re
import struct
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from uuid import uuid4

import pandas as pd
from PIL import Image
from PIL import UnidentifiedImageError
from PySide6 import QtCore, QtGui, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tagger import get_tagger, predict_tags
from backend.description_tagger import get_description_tagger, DescriptionTagResult


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ResultItem:
    path: Path
    image: Image.Image
    frame: pd.DataFrame
    caption: str


def _split_tags(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def _category_name(category: int) -> str:
    return {0: "general", 4: "character", 9: "rating"}.get(category, f"category_{category}")


def _frame_from_predictions(predictions) -> pd.DataFrame:
    rows = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "include": True,
                "rank": index + 1,
                "tag": prediction.tag,
                "confidence": round(prediction.confidence, 4),
                "category": _category_name(prediction.category),
            }
        )
    return pd.DataFrame(rows)


def _apply_filters(frame: pd.DataFrame, blacklist: Iterable[str], whitelist: Iterable[str]) -> pd.DataFrame:
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


def _sort_frame(frame: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if sort_mode == "alphabetical":
        return frame.sort_values(by=["tag", "confidence"], ascending=[True, False]).reset_index(drop=True)
    if sort_mode == "confidence":
        return frame.sort_values(by=["confidence", "tag"], ascending=[False, True]).reset_index(drop=True)
    return frame.sort_values(by=["rank", "confidence"], ascending=[True, False]).reset_index(drop=True)


def _frame_to_caption(frame: pd.DataFrame, include_scores: bool = False) -> str:
    included = frame[frame["include"]].copy()
    if included.empty:
        return ""
    included = included.sort_values(by=["rank", "confidence"], ascending=[True, False])
    if include_scores:
        return ", ".join(f"{row.tag}:{row.confidence:.3f}" for row in included.itertuples())
    return ", ".join(included["tag"].tolist())


def _export_zip(results: list[ResultItem]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            archive.writestr(f"{result.path.stem}.txt", result.caption.encode("utf-8"))
    return buffer.getvalue()


class DescriptionTagWorker(QtCore.QThread):
    """Worker thread for description-to-tags generation (prevents UI freeze)."""
    finished = QtCore.Signal(DescriptionTagResult)
    error = QtCore.Signal(str)

    def __init__(self, description: str, model: str) -> None:
        super().__init__()
        self.description = description
        self.model = model

    def run(self) -> None:
        """Run tag generation in background thread."""
        try:
            tagger = get_description_tagger(model=self.model)
            result = tagger.generate_tags(self.description)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Img-Tagboru")
        self.resize(1360, 900)
        self.setAcceptDrops(True)

        self.pending_paths: list[Path] = []
        self.results: list[ResultItem] = []
        self._active_result_index = -1
        self._preview_image: Image.Image | None = None
        self._tag_worker: DescriptionTagWorker | None = None
        self._last_description_tags: list[str] = []

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)

        # UI Styling Colors
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #4da6ff;
            }
            QPushButton {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3b3b3b;
                border: 1px solid #4da6ff;
            }
            QPushButton#tagBtn {
                background-color: #0059b3;
                font-weight: bold;
            }
            QPushButton#tagBtn:hover {
                background-color: #0073e6;
            }
            QLabel#alertLabel {
                color: #ff9933;
                font-weight: bold;
            }
        """)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs)

        # ===== TAB 1: Batch Tagger =====
        batch_tab = QtWidgets.QWidget()
        batch_layout = QtWidgets.QHBoxLayout(batch_tab)
        
        left_panel = QtWidgets.QVBoxLayout()
        right_panel = QtWidgets.QVBoxLayout()
        batch_layout.addLayout(left_panel, 1)
        batch_layout.addLayout(right_panel, 2)

        input_group = QtWidgets.QGroupBox("Input")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        button_row = QtWidgets.QHBoxLayout()
        self.open_images_btn = QtWidgets.QPushButton("Open Image")
        self.open_images_btn.clicked.connect(self.open_images)
        self.open_folder_btn = QtWidgets.QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.tag_btn = QtWidgets.QPushButton("Tag Loaded Images")
        self.tag_btn.setObjectName("tagBtn")
        self.tag_btn.clicked.connect(self.process_pending)
        button_row.addWidget(self.open_images_btn)
        button_row.addWidget(self.open_folder_btn)
        button_row.addWidget(self.tag_btn)
        input_layout.addLayout(button_row)

        # Compact drop zone hint
        drop_hint = QtWidgets.QLabel("💡 Drag images/folders or press Ctrl+V to paste")
        drop_hint.setAlignment(QtCore.Qt.AlignCenter)
        drop_hint.setStyleSheet("color: #9ecbff; font-size: 9px; margin: 5px;")
        input_layout.addWidget(drop_hint)

        # Image list label
        list_label = QtWidgets.QLabel("Loaded Images:")
        list_label.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 10px;")
        input_layout.addWidget(list_label)

        self.result_list = QtWidgets.QListWidget()
        self.result_list.currentRowChanged.connect(self.show_result)
        input_layout.addWidget(self.result_list, 1)

        self.pending_label = QtWidgets.QLabel("No images loaded.")
        self.pending_label.setObjectName("alertLabel")
        input_layout.addWidget(self.pending_label)
        left_panel.addWidget(input_group, 1)

        preview_group = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(320)
        self.image_label.setStyleSheet("background: #111; border: 1px solid #333;")
        preview_layout.addWidget(self.image_label)
        left_panel.addWidget(preview_group, 2)

        settings_group = QtWidgets.QGroupBox("Tagging Settings")
        form = QtWidgets.QFormLayout(settings_group)
        self.general_threshold = QtWidgets.QDoubleSpinBox()
        self.general_threshold.setRange(0.0, 1.0)
        self.general_threshold.setSingleStep(0.01)
        self.general_threshold.setValue(0.60)

        self.general_threshold.setStyleSheet("color: #66ff66; font-weight: bold;")

        self.character_threshold = QtWidgets.QDoubleSpinBox()
        self.character_threshold.setRange(0.0, 1.0)
        self.character_threshold.setSingleStep(0.01)
        self.character_threshold.setValue(0.85)
        self.character_threshold.setStyleSheet("color: #ff66a3; font-weight: bold;")

        self.max_tags = QtWidgets.QSpinBox()
        self.max_tags.setRange(5, 200)
        self.max_tags.setValue(40)
        self.max_tags.setStyleSheet("color: #ffcc66; font-weight: bold;")

        self.sort_mode = QtWidgets.QComboBox()
        self.sort_mode.addItems(["confidence", "alphabetical", "manual rank"])

        self.normalize_pixels = QtWidgets.QCheckBox("Normalize pixels to 0-1")
        self.use_mcut = QtWidgets.QCheckBox("Use MCut thresholding")
        self.include_scores = QtWidgets.QCheckBox("Show scores in caption")

        self.general_enabled = QtWidgets.QCheckBox("General")
        self.general_enabled.setChecked(True)
        self.character_enabled = QtWidgets.QCheckBox("Character")
        self.character_enabled.setChecked(True)

        category_widget = QtWidgets.QWidget()
        category_row = QtWidgets.QHBoxLayout(category_widget)
        category_row.setContentsMargins(0, 0, 0, 0)
        category_row.addWidget(self.general_enabled)
        category_row.addWidget(self.character_enabled)
        category_row.addStretch(1)

        self.blacklist = QtWidgets.QPlainTextEdit()
        self.blacklist.setPlaceholderText("tag1, tag2")
        self.blacklist.setFixedHeight(52)

        self.whitelist = QtWidgets.QPlainTextEdit()
        self.whitelist.setPlaceholderText("tag1, tag2 (optional)")
        self.whitelist.setFixedHeight(52)

        form.addRow("General threshold", self.general_threshold)
        form.addRow("Character threshold", self.character_threshold)
        form.addRow("Max tags", self.max_tags)
        form.addRow("Sort by", self.sort_mode)
        form.addRow("Categories", category_widget)
        form.addRow("Blacklist", self.blacklist)
        form.addRow("Whitelist", self.whitelist)
        form.addRow(self.normalize_pixels)
        form.addRow(self.use_mcut)
        form.addRow(self.include_scores)
        right_panel.addWidget(settings_group)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Include", "Rank", "Tag", "Confidence", "Category"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self.on_table_changed)
        right_panel.addWidget(self.table, 3)

        caption_group = QtWidgets.QGroupBox("Caption")
        caption_layout = QtWidgets.QVBoxLayout(caption_group)
        self.caption_edit = QtWidgets.QPlainTextEdit()
        self.caption_edit.setPlaceholderText("Selected tags will appear here.")
        caption_layout.addWidget(self.caption_edit)

        caption_buttons = QtWidgets.QHBoxLayout()
        self.apply_caption_btn = QtWidgets.QPushButton("Apply Caption Text")
        self.apply_caption_btn.clicked.connect(self.apply_caption_text)
        self.export_btn = QtWidgets.QPushButton("Save Current TXT")
        self.export_btn.clicked.connect(self.export_caption)
        self.export_all_btn = QtWidgets.QPushButton("Save All TXT")
        self.export_all_btn.clicked.connect(self.export_all_captions)
        self.export_zip_btn = QtWidgets.QPushButton("Export ZIP")
        self.export_zip_btn.clicked.connect(self.export_zip)
        caption_buttons.addWidget(self.apply_caption_btn)
        caption_buttons.addWidget(self.export_btn)
        caption_buttons.addWidget(self.export_all_btn)
        caption_buttons.addWidget(self.export_zip_btn)
        caption_layout.addLayout(caption_buttons)
        right_panel.addWidget(caption_group, 2)

        self.tabs.addTab(batch_tab, "Batch Tagger")

        # ===== TAB 2: Description Tagger =====
        desc_tab = QtWidgets.QWidget()
        desc_layout = QtWidgets.QVBoxLayout(desc_tab)

        # PC Requirements (always visible)
        req_group = QtWidgets.QGroupBox("📋 PC Requirements for Description-to-Tags")
        req_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ff9933;
                background-color: #1a1a1a;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #ff9933;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        req_layout = QtWidgets.QVBoxLayout(req_group)
        
        req_text = QtWidgets.QLabel(
            "🔹 Requires: Ollama (http://ollama.ai)\n"
            "🔹 RAM: 8-16GB minimum (16GB+ recommended)\n"
            "🔹 Disk: 10-20GB free space for model\n"
            "🔹 GPU: ⚡ HIGHLY RECOMMENDED (NVIDIA/AMD) - 10-100x faster!\n\n"
            "⚙️ Setup:\n"
            "  1. Download Ollama: ollama.ai\n"
            "  2. For GPU: Install NVIDIA drivers (CUDA) or AMD drivers (ROCm)\n"
            "  3. Run in terminal: ollama serve\n"
            "  4. Check GPU usage: ollama ps (shows 'GPU loaded')\n"
            "  5. Pull model: ollama pull qwen2:7b\n"
            "  6. Return here and generate tags!\n\n"
            "💡 Tip: Without GPU, 14B models take 5-30 minutes. With GPU: 10-60 seconds."
        )
        req_text.setWordWrap(True)
        req_text.setStyleSheet("color: #ffcc66; font-family: monospace; font-size: 9px;")
        req_layout.addWidget(req_text)
        desc_layout.addWidget(req_group)

        # Model selection
        model_group = QtWidgets.QGroupBox("Model Selection")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        
        model_label = QtWidgets.QLabel("Select LLM Model:")
        model_label.setStyleSheet("color: #4da6ff; font-size: 9px;")
        model_layout.addWidget(model_label)
        
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        self.model_selector.addItem("(Loading models...)", None)
        model_layout.addWidget(self.model_selector)
        self._refresh_available_models()
        
        refresh_models_btn = QtWidgets.QPushButton("Refresh Available Models")
        refresh_models_btn.clicked.connect(self._refresh_available_models)
        model_layout.addWidget(refresh_models_btn)
        
        desc_layout.addWidget(model_group)

        # Description input section
        input_desc_group = QtWidgets.QGroupBox("Description Input")
        input_desc_layout = QtWidgets.QVBoxLayout(input_desc_group)
        
        desc_hint = QtWidgets.QLabel("Describe what you want to see in the image:")
        desc_hint.setStyleSheet("color: #4da6ff; font-size: 9px;")
        input_desc_layout.addWidget(desc_hint)
        
        self.description_input = QtWidgets.QPlainTextEdit()
        self.description_input.setPlaceholderText(
            "Examples:\n"
            "• girl with long black hair, red eyes, wearing maid outfit\n"
            "• anime boy, blue eyes, blonde hair, holding sword\n"
            "• landscape with mountains, sunset, fantasy art style"
        )
        self.description_input.setMinimumHeight(150)
        self.description_input.setStyleSheet("background-color: #1a1a1a; color: #ffffff; border: 1px solid #444;")
        input_desc_layout.addWidget(self.description_input)
        
        self.generate_from_desc_btn = QtWidgets.QPushButton("Generate Tags from Description")
        self.generate_from_desc_btn.setObjectName("tagBtn")
        self.generate_from_desc_btn.setMinimumHeight(40)
        self.generate_from_desc_btn.setStyleSheet("""
            QPushButton#tagBtn {
                background-color: #0059b3;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton#tagBtn:hover {
                background-color: #0073e6;
            }
        """)
        self.generate_from_desc_btn.clicked.connect(self._generate_tags_from_description)
        input_desc_layout.addWidget(self.generate_from_desc_btn)
        
        desc_layout.addWidget(input_desc_group)

        # Generated tags display
        tags_group = QtWidgets.QGroupBox("Generated Tags")
        tags_layout = QtWidgets.QVBoxLayout(tags_group)
        
        self.desc_tags_display = QtWidgets.QPlainTextEdit()
        self.desc_tags_display.setReadOnly(True)
        self.desc_tags_display.setPlaceholderText("Generated tags will appear here after processing...")
        self.desc_tags_display.setStyleSheet("background-color: #1a1a1a; color: #66ff66; font-family: monospace;")
        tags_layout.addWidget(self.desc_tags_display)
        
        # Copy button for description tags
        copy_tags_btn = QtWidgets.QPushButton("📋 Copy Tags (Comma-Separated)")
        copy_tags_btn.setStyleSheet("background-color: #ff9933; color: white; font-weight: bold;")
        copy_tags_btn.clicked.connect(self._copy_description_tags)
        self._copy_tags_btn = copy_tags_btn
        tags_layout.addWidget(copy_tags_btn)
        
        desc_layout.addWidget(tags_group)
        
        self.tabs.addTab(desc_tab, "Description Tagger")

        self.statusbar = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(220)
        self.statusbar.addPermanentWidget(self.progress)

        self._set_export_enabled(False)

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)
        self.export_all_btn.setEnabled(enabled)
        self.export_zip_btn.setEnabled(enabled)

    def _selected_categories(self) -> set[str]:
        selected = set()
        if self.general_enabled.isChecked():
            selected.add("general")
        if self.character_enabled.isChecked():
            selected.add("character")
        return selected

    def _load_paths(self, paths: list[Path]) -> None:
        valid_paths: list[Path] = []
        skipped = 0
        for path in paths:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = self._open_image_safe(path)
            if image is None:
                skipped += 1
                continue
            valid_paths.append(path)

        # Clear previous results and tags when loading new images
        self.results = []
        self._active_result_index = -1
        self.table.setRowCount(0)
        self.caption_edit.setPlainText("")
        self._set_export_enabled(False)
        
        self.pending_paths = valid_paths
        self.pending_label.setText(f"Loaded {len(self.pending_paths)} image(s).")
        self.tag_btn.setEnabled(bool(self.pending_paths))
        # populate the left list so the user can preview any image before tagging
        self.result_list.blockSignals(True)
        self.result_list.clear()
        for p in self.pending_paths:
            self.result_list.addItem(p.name)
        self.result_list.blockSignals(False)
        if self.pending_paths:
            # show first image preview immediately with cleared tags
            self.result_list.setCurrentRow(0)
            self.show_result(0)  # Explicitly show first pending image with cleared tags
        elif skipped:
            self.statusbar.showMessage("No valid images found in drop/paste input.", 7000)

    def _open_image_safe(self, path: Path) -> Image.Image | None:
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

    def open_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if paths:
            self._load_paths([Path(path) for path in paths])

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        # Accept drops for: URLs, images, HTML, text, or any image MIME types
        if (
            mime_data.hasUrls()
            or mime_data.hasImage()
            or mime_data.hasHtml()
            or mime_data.hasText()
            or any(fmt.startswith("image/") for fmt in mime_data.formats())
            or mime_data.hasFormat("application/octet-stream")
        ):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def _save_qimage_temp(self, image: QtGui.QImage, prefix: str) -> Path | None:
        if image.isNull():
            return None
        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-clipboard"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{prefix}_{uuid4().hex[:8]}.png"
        if image.save(str(temp_path), "PNG"):
            return temp_path
        return None

    def _download_web_image_to_temp(self, url: str) -> Path | None:
        try:
            print(f"[DEBUG] Starting download of: {url}")
            
            # Try with increasingly realistic browser headers
            headers_variants = [
                # Full Chrome-like headers
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://danbooru.donmai.us/",
                    "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Sec-Fetch-Dest": "image",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "same-site",
                    "Cache-Control": "max-age=0",
                },
                # Minimal headers
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                # No special headers
                {},
            ]
            
            for attempt, headers in enumerate(headers_variants):
                try:
                    print(f"[DEBUG] Attempt {attempt + 1} with headers: {list(headers.keys())}")
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response:
                        content_type = (response.headers.get("Content-Type") or "").lower()
                        data = response.read()
                        print(f"[DEBUG] Downloaded {len(data)} bytes, Content-Type: {content_type}")

                        # Be lenient: accept image types, application/octet-stream, or if PIL can open it
                        if content_type and content_type not in ("application/octet-stream", "") and "image/" not in content_type:
                            print(f"[DEBUG] Rejected due to Content-Type: {content_type}")
                            continue

                        # Validate with PIL before accepting as a dropped image
                        try:
                            img = Image.open(io.BytesIO(data))
                            img.load()  # Force load to validate
                            print(f"[DEBUG] PIL validation successful, format: {img.format}")
                        except (UnidentifiedImageError, OSError) as e:
                            print(f"[DEBUG] PIL validation failed: {e}")
                            continue

                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        # Use correct extension based on format or default to jpg
                        ext = (img.format or "jpg").lower() if hasattr(img, 'format') else "jpg"
                        ext = ext if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif") else "jpg"
                        temp_path = temp_dir / f"web_{uuid4().hex[:8]}.{ext}"
                        temp_path.write_bytes(data)
                        print(f"[DEBUG] Saved to {temp_path}")
                        return temp_path
                except urllib.error.HTTPError as e:
                    print(f"[DEBUG] HTTP Error {e.code} on attempt {attempt + 1}")
                    if attempt == len(headers_variants) - 1:
                        # Last attempt failed
                        raise
                    continue
                except Exception as e:
                    print(f"[DEBUG] Error on attempt {attempt + 1}: {e}")
                    if attempt == len(headers_variants) - 1:
                        raise
                    continue

        except Exception as e:
            print(f"[DEBUG] Download failed with exception: {type(e).__name__}: {e}")
            return None

    def _extract_web_image_candidates(self, mime_data: QtCore.QMimeData) -> list[str]:
        candidates: list[str] = []

        if mime_data.hasUrls():
            for url in mime_data.urls():
                if url.scheme() in ("http", "https"):
                    candidates.append(url.toString())

        if mime_data.hasHtml():
            html = mime_data.html()
            src_matches = re.findall(r'src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
            base_match = re.search(r'<base[^>]*href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
            base_url = base_match.group(1) if base_match else ""
            for src in src_matches:
                if src.startswith(("http://", "https://")):
                    candidates.append(src)
                elif base_url:
                    candidates.append(urljoin(base_url, src))

        if mime_data.hasText():
            text = mime_data.text().strip()
            if text.startswith(("http://", "https://")):
                candidates.append(text)

        # De-duplicate while preserving order.
        unique: list[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime_data = event.mimeData()

        # 1) Local files/folders from Explorer.
        if mime_data.hasUrls():
            urls = mime_data.urls()
            local_paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
            if local_paths:
                files = []
                for path in local_paths:
                    if path.is_dir():
                        files.extend([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
                    else:
                        files.append(path)
                        
                if files:
                    self._load_paths(files)
                    self.statusbar.showMessage(f"Loaded {len(files)} dropped file(s).", 5000)
                    event.acceptProposedAction()
                    return

        # 2) Try Qt's native image support first (handles DragImageBits natively)
        if mime_data.hasImage():
            print(f"[DEBUG] Attempting to use Qt's native image support via hasImage()")
            try:
                image = QtGui.QImage(mime_data.imageData())
                if not image.isNull():
                    print(f"[DEBUG] Successfully got QImage from mime_data")
                    temp_path = self._save_qimage_temp(image, "dragged")
                    if temp_path is not None:
                        print(f"[DEBUG] Saved QImage to {temp_path}")
                        self._load_paths([temp_path])
                        self.statusbar.showMessage("Loaded dropped image.", 5000)
                        event.acceptProposedAction()
                        return
            except Exception as e:
                print(f"[DEBUG] Qt image handling failed: {e}")

        # 3) Try to extract raw image data from ANY available MIME type
        print(f"[DEBUG] Available MIME formats: {mime_data.formats()}")
        for fmt in mime_data.formats():
            print(f"[DEBUG] Trying to extract image from format: {fmt}")
            try:
                image_bytes = mime_data.data(fmt)
                if image_bytes and len(image_bytes) > 500:  # Real image data should be substantial
                    print(f"[DEBUG] Found {len(image_bytes)} bytes in format {fmt}")
                    
                    # Try Qt's QImage.fromData() which auto-detects format
                    print(f"[DEBUG] Attempting QImage.fromData() on {fmt}")
                    qt_image = QtGui.QImage.fromData(image_bytes)
                    if not qt_image.isNull():
                        print(f"[DEBUG] QImage.fromData() succeeded for {fmt}")
                        temp_path = self._save_qimage_temp(qt_image, "dragged")
                        if temp_path is not None:
                            print(f"[DEBUG] Saved QImage to {temp_path}")
                            self._load_paths([temp_path])
                            self.statusbar.showMessage("Loaded dropped image.", 5000)
                            event.acceptProposedAction()
                            return
                    
                    # Standard PIL image format handling
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        img.load()  # Verify it's valid
                        print(f"[DEBUG] Successfully parsed as {img.format}")
                        
                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        ext = (img.format or "jpg").lower() if hasattr(img, 'format') else "jpg"
                        ext = ext if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif") else "jpg"
                        temp_path = temp_dir / f"web_{uuid4().hex[:8]}.{ext}"
                        temp_path.write_bytes(image_bytes)
                        print(f"[DEBUG] Saved extracted image to {temp_path}")
                        self._load_paths([temp_path])
                        self.statusbar.showMessage("Loaded dropped image.", 5000)
                        event.acceptProposedAction()
                        return
                    except Exception as e:
                        print(f"[DEBUG] Failed to parse {fmt} as PIL image: {e}")
                        continue
            except Exception as e:
                print(f"[DEBUG] Error extracting {fmt}: {e}")
                continue

        # 4) Web URLs from browsers (url/html/text).
        candidates = list(self._extract_web_image_candidates(mime_data))
        if candidates:
            print(f"[DEBUG] Found web image candidates: {candidates}")
        for url in candidates:
            print(f"[DEBUG] Downloading web image from URL: {url}")
            downloaded = self._download_web_image_to_temp(url)
            if downloaded and downloaded.suffix.lower() in IMAGE_EXTENSIONS | {".png"}:
                print(f"[DEBUG] Successfully downloaded to {downloaded}")
                self._load_paths([downloaded])
                self.statusbar.showMessage("Loaded dropped image from web URL.", 5000)
                event.acceptProposedAction()
                return

        # Debug: Show what MIME types we got
        all_formats = mime_data.formats()
        print(f"[DEBUG] Drop not recognized. Available MIME types: {all_formats}")
        self.statusbar.showMessage(
            "Drop not recognized. Try: drag image file, drag from website, or Ctrl+V.",
            7000,
        )

        super().dropEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            clipboard = QtWidgets.QApplication.clipboard()
            mime_data = clipboard.mimeData()

            # First try URLs (copying a file from Explorer)
            if mime_data.hasUrls():
                paths = [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()]
                if paths:
                    self._load_paths(paths)
                    self.statusbar.showMessage("Loaded pasted file path(s).", 5000)
                    event.accept()
                    return

            # Next try URLs copied from browser.
            for url in self._extract_web_image_candidates(mime_data):
                downloaded = self._download_web_image_to_temp(url)
                if downloaded is not None:
                    self._load_paths([downloaded])
                    self.statusbar.showMessage("Loaded pasted web image URL.", 5000)
                    event.accept()
                    return

            # Next try an image from the clipboard (like copying from Snipping Tool or web browser)
            image = clipboard.image()
            temp_path = self._save_qimage_temp(image, "clipboard")
            if temp_path is not None:
                self._load_paths([temp_path])
                self.statusbar.showMessage("Loaded pasted image data.", 5000)
                event.accept()
                return

            self.statusbar.showMessage(
                "Clipboard does not contain an image. Copy an image or an image URL and press Ctrl+V.",
                7000,
            )

        super().keyPressEvent(event)

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
        if not folder:
            return
        root = Path(folder)
        paths = [path for path in sorted(root.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
        self._load_paths(paths)

    def _tag_image(self, image: Image.Image):
        tagger = get_tagger()
        return predict_tags(
            tagger,
            image,
            general_threshold=self.general_threshold.value(),
            character_threshold=self.character_threshold.value(),
            normalize_pixels=self.normalize_pixels.isChecked(),
            use_mcut=self.use_mcut.isChecked(),
            limit=self.max_tags.value(),
        )

    def process_pending(self) -> None:
        if not self.pending_paths:
            return

        self.results = []
        self._active_result_index = -1
        # keep the result_list (names) populated so previews remain selectable
        self.table.blockSignals(True)
        self.result_list.blockSignals(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusbar.showMessage("Tagging images...")

        blacklist = _split_tags(self.blacklist.toPlainText())
        whitelist = _split_tags(self.whitelist.toPlainText())
        allowed_categories = self._selected_categories()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            total = len(self.pending_paths)
            for index, path in enumerate(self.pending_paths, start=1):
                image = self._open_image_safe(path)
                if image is None:
                    self.progress.setValue(int(index / max(1, total) * 100))
                    QtWidgets.QApplication.processEvents()
                    continue
                predictions = self._tag_image(image)
                frame = _frame_from_predictions(predictions)
                if not frame.empty:
                    if allowed_categories:
                        frame = frame[frame["category"].isin(allowed_categories)].copy()
                    frame = _apply_filters(frame, blacklist, whitelist)
                    frame = _sort_frame(frame, self.sort_mode.currentText())
                caption = _frame_to_caption(frame, include_scores=self.include_scores.isChecked())
                self.results.append(ResultItem(path=path, image=image, frame=frame, caption=caption))
                # mark the existing list item as tagged (avoid duplicating entries)
                try:
                    item = self.result_list.item(index - 1)
                    if item is not None:
                        item.setText(path.name)
                except Exception:
                    pass
                self.progress.setValue(int(index / max(1, total) * 100))
                QtWidgets.QApplication.processEvents()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setVisible(False)
            self.table.blockSignals(False)
            self.result_list.blockSignals(False)

        self._set_export_enabled(bool(self.results))
        self.statusbar.showMessage(f"Tagged {len(self.results)} image(s).")
        if self.results:
            current_row = self.result_list.currentRow()
            if current_row >= 0:
                self.show_result(current_row)
            else:
                self.result_list.setCurrentRow(0)

    def _current_index(self) -> int:
        return self.result_list.currentRow()

    def _current_result(self) -> ResultItem | None:
        index = self._current_index()
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def _sync_current_result(self) -> None:
        result = self._current_result()
        if result is None:
            return
        result.frame = self._table_to_frame()
        result.caption = self.caption_edit.toPlainText().strip()

    def show_result(self, index: int) -> None:
        if not (0 <= index < len(self.results)):
            # if index is within pending_paths, show the preview image
            if 0 <= index < len(self.pending_paths):
                self._active_result_index = -1
                img = self._open_image_safe(self.pending_paths[index])
                if img is not None:
                    self._set_image(img)
                else:
                    self.image_label.clear()
                    self.statusbar.showMessage("Could not preview this image file.", 7000)
                self.table.setRowCount(0)
                self.caption_edit.setPlainText("")
            return

        # save edits from previously active result (if any)
        if 0 <= self._active_result_index < len(self.results) and self.table.rowCount() > 0:
            previous_result = self.results[self._active_result_index]
            previous_result.frame = self._table_to_frame()
            previous_result.caption = self.caption_edit.toPlainText().strip()

        result = self.results[index]
        self._active_result_index = index
        self._set_image(result.image)
        self._frame_to_table(result.frame)
        self.caption_edit.setPlainText(result.caption)

    def _set_image(self, pil: Image.Image) -> None:
        self._preview_image = pil
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        pixmap = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(buf.getvalue()))
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._preview_image is not None:
            self._set_image(self._preview_image)

    def _frame_to_table(self, frame: pd.DataFrame) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row_index, row in enumerate(frame.itertuples(index=False)):
            self.table.insertRow(row_index)

            include_item = QtWidgets.QTableWidgetItem()
            include_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            include_item.setCheckState(QtCore.Qt.Checked if bool(row.include) else QtCore.Qt.Unchecked)
            self.table.setItem(row_index, 0, include_item)

            rank_item = QtWidgets.QTableWidgetItem(str(int(row.rank)))
            rank_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)
            self.table.setItem(row_index, 1, rank_item)

            tag_item = QtWidgets.QTableWidgetItem(str(row.tag))
            tag_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 2, tag_item)

            confidence_item = QtWidgets.QTableWidgetItem(f"{float(row.confidence):.4f}")
            confidence_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 3, confidence_item)

            category_item = QtWidgets.QTableWidgetItem(str(row.category))
            category_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row_index, 4, category_item)
        self.table.blockSignals(False)

    def _table_to_frame(self) -> pd.DataFrame:
        rows = []
        for row_index in range(self.table.rowCount()):
            include_item = self.table.item(row_index, 0)
            rank_item = self.table.item(row_index, 1)
            tag_item = self.table.item(row_index, 2)
            confidence_item = self.table.item(row_index, 3)
            category_item = self.table.item(row_index, 4)

            rows.append(
                {
                    "include": include_item.checkState() == QtCore.Qt.Checked if include_item else False,
                    "rank": int(rank_item.text()) if rank_item and rank_item.text().strip().isdigit() else row_index + 1,
                    "tag": tag_item.text() if tag_item else "",
                    "confidence": float(confidence_item.text()) if confidence_item else 0.0,
                    "category": category_item.text() if category_item else "",
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = _sort_frame(frame, self.sort_mode.currentText())
        return frame

    def on_table_changed(self, *_args) -> None:
        result = self._current_result()
        if result is None:
            return
        result.frame = self._table_to_frame()
        result.caption = _frame_to_caption(result.frame, include_scores=self.include_scores.isChecked())
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(result.caption)
        self.caption_edit.blockSignals(False)

    def apply_caption_text(self) -> None:
        result = self._current_result()
        if result is None:
            return
        tags = _split_tags(self.caption_edit.toPlainText())
        tag_order = {tag.lower(): index + 1 for index, tag in enumerate(tags)}
        frame = result.frame.copy()
        lower_tags = {tag.lower() for tag in tags}
        frame["include"] = frame["tag"].astype(str).str.lower().isin(lower_tags)
        frame["rank"] = [tag_order.get(str(tag).lower(), 9999) for tag in frame["tag"]]
        frame = _sort_frame(frame, self.sort_mode.currentText())
        result.frame = frame
        result.caption = _frame_to_caption(frame, include_scores=self.include_scores.isChecked())
        self._frame_to_table(frame)
        self.caption_edit.setPlainText(result.caption)

    def _show_pc_requirements_warning(self) -> None:
        """Show information about PC requirements for running LLM models locally."""
        # Deprecated - PC requirements now shown prominently in Description Tagger tab
        pass

    def _refresh_available_models(self) -> None:
        """Refresh the list of available Ollama models."""
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        try:
            tagger = get_description_tagger()
            if not tagger.check_connection():
                self.model_selector.addItem("(Ollama not running)", None)
            else:
                models = tagger.list_available_models()
                for model in models:
                    self.model_selector.addItem(model, model)
                preferred_model = next(
                    (
                        model
                        for model in models
                        if "qwen3-14b" in model.lower() or "14b" in model.lower()
                    ),
                    models[0] if models else None,
                )
                if preferred_model:
                    index = self.model_selector.findData(preferred_model)
                    if index >= 0:
                        self.model_selector.setCurrentIndex(index)
        except:
            self.model_selector.addItem("(error loading)", None)
        finally:
            self.model_selector.blockSignals(False)

    def _generate_tags_from_description(self) -> None:
        """Generate Danbooru tags from text description using Ollama (non-blocking)."""
        description = self.description_input.toPlainText().strip()
        if not description:
            self.statusbar.showMessage("Description is empty. Please enter a description.", 5000)
            return

        selected_model = self.model_selector.currentData()
        if not selected_model:
            self.desc_tags_display.setPlainText("⚠️ Error:\n\nNo model selected. Refresh models or start Ollama.")
            self.statusbar.showMessage("Tag generation failed.", 5000)
            return

        # Disable button and show loading state
        self.generate_from_desc_btn.setEnabled(False)
        self.desc_tags_display.setPlainText("⏳ Generating tags... (this may take a while)\n\n⚠️ TIP: Enable GPU in Ollama for faster generation!")
        self.statusbar.showMessage("Connecting to Ollama and generating tags...")

        # Create and start worker thread
        self._tag_worker = DescriptionTagWorker(description, selected_model)
        self._tag_worker.finished.connect(self._on_tags_generated)
        self._tag_worker.error.connect(self._on_tag_generation_error)
        self._tag_worker.start()

    def _on_tags_generated(self, result: DescriptionTagResult) -> None:
        """Handle successful tag generation."""
        # Store tags for copy button
        self._last_description_tags = result.tags
        
        tags_output = ", ".join(result.tags)
        
        if len(result.tags) == 0:
            display_text = (
                "⚠️ No output generated\n\n"
                "Try refining your description with more visual details."
            )
            self._copy_tags_btn.setEnabled(False)
        else:
            display_text = f"✓ Generated prompt ({len(result.tags)} terms):\n\n{tags_output}"
            self._copy_tags_btn.setEnabled(True)
        
        self.desc_tags_display.setPlainText(display_text)
        self.statusbar.showMessage(f"✓ Generated {len(result.tags)} prompt terms.", 5000)
        self.generate_from_desc_btn.setEnabled(True)
        self._tag_worker = None

    def _on_tag_generation_error(self, error_msg: str) -> None:
        """Handle tag generation error."""
        self.desc_tags_display.setPlainText(f"⚠️ Error:\n\n{error_msg}")
        self.statusbar.showMessage("Tag generation failed.", 5000)
        self.generate_from_desc_btn.setEnabled(True)
        self._copy_tags_btn.setEnabled(False)
        self._tag_worker = None

    def _copy_description_tags(self) -> None:
        """Copy generated description tags to clipboard as comma-separated list."""
        if not hasattr(self, '_last_description_tags') or not self._last_description_tags:
            self.statusbar.showMessage("No tags to copy.", 2000)
            return
        
        tags_text = ", ".join(self._last_description_tags)
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(tags_text)
        self.statusbar.showMessage(f"✓ Copied {len(self._last_description_tags)} tags to clipboard", 3000)

    def export_caption(self) -> None:
        result = self._current_result()
        if result is None:
            return
        self._sync_current_result()
        default_path = str(result.path.with_suffix(".txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save caption", default_path, "Text (*.txt)")
        if not path:
            return
        Path(path).write_text(result.caption, encoding="utf-8")
        self.statusbar.showMessage(f"Saved caption to {path}")

    def export_all_captions(self) -> None:
        if not self.results:
            return
        self._sync_current_result()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder", str(Path.cwd()))
        if not folder:
            return
        output = Path(folder)
        for result in self.results:
            (output / f"{result.path.stem}.txt").write_text(result.caption, encoding="utf-8")
        self.statusbar.showMessage(f"Saved {len(self.results)} caption files.")

    def export_zip(self) -> None:
        if not self.results:
            return
        self._sync_current_result()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save captions zip",
            str(Path.cwd() / "captions.zip"),
            "Zip (*.zip)",
        )
        if not path:
            return
        Path(path).write_bytes(_export_zip(self.results))
        self.statusbar.showMessage(f"Saved zip to {path}")


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
