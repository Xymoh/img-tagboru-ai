from __future__ import annotations

import io
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin
from uuid import uuid4

import pandas as pd
from PIL import Image, UnidentifiedImageError
from PySide6 import QtCore, QtGui, QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.description_tagger import DescriptionTagResult, get_description_tagger
from backend.tagger import category_label, get_tagger, predict_tags
from backend.tag_utils import (
    IMAGE_EXTENSIONS,
    TaggingResult,
    apply_filters,
    export_zip_from_results,
    frame_from_predictions,
    frame_to_caption,
    sort_frame,
    split_tags,
)

from frontend.native.completer import CaptionCompleterMixin
from frontend.native.styles import build_stylesheet
from frontend.native.widgets import HelpDialog
from frontend.native.workers import DescriptionTagWorker


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow, CaptionCompleterMixin):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Img-Tagboru v1.2")
        self.resize(1400, 300)
        self.setAcceptDrops(True)

        self.pending_paths: list[Path] = []
        self.results: list[TaggingResult] = []
        self._single_results: dict[int, TaggingResult] = {}
        self._active_result_index = -1
        self._preview_image: Image.Image | None = None
        self._tag_worker: DescriptionTagWorker | None = None
        self._last_description_tags: list[str] = []
        self._last_creativity_mode = "creative"
        self.danbooru_tags: list[str] = []
        self.caption_completer: QtWidgets.QCompleter | None = None

        # Load Danbooru tags for autocomplete
        self.danbooru_tags = self._load_danbooru_tags()

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(10, 10, 10, 10)

        self.setStyleSheet(build_stylesheet())

        # ── Tab widget ────────────────────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(self.tabs)

        # ===== TAB 1: Batch Tagger =====
        batch_tab = QtWidgets.QWidget()
        batch_layout = QtWidgets.QHBoxLayout(batch_tab)
        batch_layout.setSpacing(5)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(5)
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(5)
        batch_layout.addLayout(left_panel, 1)
        batch_layout.addLayout(right_panel, 2)

        input_group = QtWidgets.QGroupBox("📁 Input - Load Images")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        input_layout.setSpacing(8)

        button_row = QtWidgets.QHBoxLayout()
        self.open_images_btn = QtWidgets.QPushButton("🖼️ Open Images")
        self.open_images_btn.setToolTip("Select one or more image files to tag")
        self.open_images_btn.clicked.connect(self.open_images)
        self.open_folder_btn = QtWidgets.QPushButton("📂 Open Folder")
        self.open_folder_btn.setToolTip("Select a folder containing images\n(Can include subfolders on user choice)")
        self.open_folder_btn.clicked.connect(self.open_folder)
        self.tag_selected_btn = QtWidgets.QPushButton("⚡ Tag Selected")
        self.tag_selected_btn.setObjectName("tagSelectedBtn")
        self.tag_selected_btn.setToolTip("Tag only the currently selected image")
        self.tag_selected_btn.clicked.connect(self.process_single_image)
        self.tag_selected_btn.setEnabled(False)
        self.tag_btn = QtWidgets.QPushButton("🏷️ Tag All Images")
        self.tag_btn.setObjectName("tagBtn")
        self.tag_btn.setToolTip("Start tagging all loaded images with current settings")
        self.tag_btn.clicked.connect(self.process_pending)
        button_row.addWidget(self.open_images_btn)
        button_row.addWidget(self.open_folder_btn)
        button_row.addWidget(self.tag_selected_btn)
        button_row.addWidget(self.tag_btn)
        input_layout.addLayout(button_row)

        drop_hint = QtWidgets.QLabel(
            "💡 Tip: Drag & drop images/folders here, or press Ctrl+V to paste from clipboard"
        )
        drop_hint.setAlignment(QtCore.Qt.AlignCenter)
        drop_hint.setStyleSheet(
            "color: #9ecbff; font-size: 10px; margin: 8px; padding: 8px; "
            "background-color: #0d0d0d; border: 1px dashed #444; border-radius: 5px;"
        )
        input_layout.addWidget(drop_hint)

        list_label = QtWidgets.QLabel("📋 Loaded Images:")
        list_label.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        input_layout.addWidget(list_label)

        self.result_list = QtWidgets.QListWidget()
        self.result_list.currentRowChanged.connect(self.show_result)
        self.result_list.currentRowChanged.connect(self._update_tag_selected_button)
        self.result_list.setToolTip("Click an image to preview and edit its tags")
        input_layout.addWidget(self.result_list, 1)

        self.pending_label = QtWidgets.QLabel(
            "ℹ️ No images loaded. Use buttons above or drag & drop files."
        )
        self.pending_label.setObjectName("alertLabel")
        self.pending_label.setWordWrap(True)
        input_layout.addWidget(self.pending_label)
        left_panel.addWidget(input_group, 1)

        preview_group = QtWidgets.QGroupBox("🖼️ Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(250)
        self.image_label.setMaximumHeight(350)
        self.image_label.setStyleSheet(
            "background: #0d0d0d; border: 2px solid #333; border-radius: 8px; padding: 2px;"
        )
        self.image_label.setToolTip("Preview of selected image")
        preview_layout.addWidget(self.image_label)
        left_panel.addWidget(preview_group, 1)

        settings_group = QtWidgets.QGroupBox("⚙️ Tagging Settings")
        form = QtWidgets.QFormLayout(settings_group)
        form.setSpacing(8)

        self.general_threshold = QtWidgets.QDoubleSpinBox()
        self.general_threshold.setRange(0.0, 1.0)
        self.general_threshold.setSingleStep(0.01)
        self.general_threshold.setValue(0.6)
        self.general_threshold.setToolTip(
            "Lower = more tags (0.25-0.40 recommended)\nHigher = only very confident tags"
        )
        self.general_threshold.setStyleSheet("color: #66ff66; font-weight: bold;")

        self.character_threshold = QtWidgets.QDoubleSpinBox()
        self.character_threshold.setRange(0.0, 1.0)
        self.character_threshold.setSingleStep(0.01)
        self.character_threshold.setValue(0.85)
        self.character_threshold.setToolTip(
            "Higher = only confident character matches (0.80-0.95)\nLower = may detect false characters"
        )
        self.character_threshold.setStyleSheet("color: #ff66a3; font-weight: bold;")

        self.max_tags = QtWidgets.QSpinBox()
        self.max_tags.setRange(5, 200)
        self.max_tags.setValue(40)
        self.max_tags.setToolTip("Limit tags per image (40-80 is typical for training)")
        self.max_tags.setStyleSheet("color: #ffcc66; font-weight: bold;")

        self.sort_mode = QtWidgets.QComboBox()
        self.sort_mode.addItems(["confidence", "alphabetical", "manual rank"])
        self.sort_mode.setToolTip("How to order tags in the results table")

        self.normalize_pixels = QtWidgets.QCheckBox("Normalize pixels to 0-1")
        self.normalize_pixels.setToolTip("Standardize pixel values (usually not needed for WD14)")
        self.use_mcut = QtWidgets.QCheckBox("Use MCut thresholding")
        self.use_mcut.setToolTip("Automatic threshold detection (overrides manual thresholds)")
        self.include_scores = QtWidgets.QCheckBox("Show scores in caption")
        self.include_scores.setToolTip("Include confidence scores in exported captions")

        self.general_enabled = QtWidgets.QCheckBox("General tags")
        self.general_enabled.setChecked(True)
        self.general_enabled.setToolTip("Include general tags (clothing, background, etc.)")
        self.character_enabled = QtWidgets.QCheckBox("Character tags")
        self.character_enabled.setChecked(True)
        self.character_enabled.setToolTip("Include character name tags")

        category_widget = QtWidgets.QWidget()
        category_row = QtWidgets.QHBoxLayout(category_widget)
        category_row.setContentsMargins(0, 0, 0, 0)
        category_row.addWidget(self.general_enabled)
        category_row.addWidget(self.character_enabled)
        category_row.addStretch(1)

        self.blacklist = QtWidgets.QPlainTextEdit()
        self.blacklist.setPlaceholderText("blurry, lowres, bad_anatomy")
        self.blacklist.setFixedHeight(52)
        self.blacklist.setToolTip("Tags to always exclude (comma-separated)")

        self.whitelist = QtWidgets.QPlainTextEdit()
        self.whitelist.setPlaceholderText("1girl, solo (optional)")
        self.whitelist.setFixedHeight(52)
        self.whitelist.setToolTip("Only include these tags if specified (comma-separated)")

        form.addRow(QtWidgets.QLabel("General threshold:"), self.general_threshold)
        form.addRow(QtWidgets.QLabel("Character threshold:"), self.character_threshold)
        form.addRow(QtWidgets.QLabel("Max tags:"), self.max_tags)
        form.addRow(QtWidgets.QLabel("Sort by:"), self.sort_mode)
        form.addRow(QtWidgets.QLabel("Categories:"), category_widget)
        form.addRow(QtWidgets.QLabel("Blacklist:"), self.blacklist)
        form.addRow(QtWidgets.QLabel("Whitelist:"), self.whitelist)
        form.addRow(self.normalize_pixels)
        form.addRow(self.use_mcut)
        form.addRow(self.include_scores)
        right_panel.addWidget(settings_group)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["✓ Include", "Rank", "Tag", "Confidence", "Category"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip(
            "Uncheck 'Include' to exclude tags from caption\nEdit 'Rank' to change tag order"
        )
        self.table.itemChanged.connect(self.on_table_changed)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(2, 350)
        self.table.setColumnWidth(4, 90)
        right_panel.addWidget(self.table, 3)

        caption_group = QtWidgets.QGroupBox("📝 Generated Caption")
        caption_layout = QtWidgets.QVBoxLayout(caption_group)
        self.caption_edit = QtWidgets.QPlainTextEdit()
        self.caption_edit.setPlaceholderText(
            "Tags will appear here as comma-separated values...\n"
            "Example: 1girl, smile, blue_eyes, long_hair"
        )
        self.caption_edit.setToolTip("Edit caption text directly, then click 'Apply' to sync with table")
        caption_layout.addWidget(self.caption_edit)

        caption_buttons = QtWidgets.QHBoxLayout()
        self.apply_caption_btn = QtWidgets.QPushButton("🔄 Apply Caption")
        self.apply_caption_btn.setToolTip("Update table from edited caption text")
        self.apply_caption_btn.clicked.connect(self.apply_caption_text)
        self.export_btn = QtWidgets.QPushButton("💾 Save Current")
        self.export_btn.setToolTip("Save caption for selected image as .txt file")
        self.export_btn.clicked.connect(self.export_caption)
        self.export_all_btn = QtWidgets.QPushButton("💾 Save All")
        self.export_all_btn.setToolTip("Save all captions to a folder")
        self.export_all_btn.clicked.connect(self.export_all_captions)
        self.export_zip_btn = QtWidgets.QPushButton("📦 Export ZIP")
        self.export_zip_btn.setToolTip("Download all captions as ZIP file")
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

        model_group = QtWidgets.QGroupBox("🤖 LLM Model Selection")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        model_label = QtWidgets.QLabel("Select AI Model for Tag Generation:")
        model_label.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        model_layout.addWidget(model_label)

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 5px;"
        )
        self.model_selector.addItem("(Loading models...)", None)
        model_layout.addWidget(self.model_selector)
        self._refresh_available_models()

        refresh_models_btn = QtWidgets.QPushButton("🔄 Refresh Models")
        refresh_models_btn.setToolTip("Check for newly installed Ollama models")
        refresh_models_btn.clicked.connect(self._refresh_available_models)
        model_layout.addWidget(refresh_models_btn)

        desc_layout.addWidget(model_group)

        creativity_group = QtWidgets.QGroupBox("🎨 Creativity Mode")
        creativity_layout = QtWidgets.QVBoxLayout(creativity_group)
        creativity_layout.setSpacing(8)

        creativity_label = QtWidgets.QLabel("How imaginative should the generated tags be?")
        creativity_label.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        creativity_layout.addWidget(creativity_label)

        self.creativity_selector = QtWidgets.QComboBox()
        self.creativity_selector.setStyleSheet(
            "background-color: #1a1a1a; color: #ffffff; padding: 5px;"
        )
        self.creativity_selector.addItem("🛡️ Safe (literal, conservative)", "safe")
        self.creativity_selector.addItem("✨ Creative (balanced)", "creative")
        self.creativity_selector.addItem("🔞 Mature (suggestive, nsfw)", "mature")
        self.creativity_selector.addItem("💀 Extreme (wild ideas)", "extreme")
        self.creativity_selector.setCurrentIndex(1)
        creativity_layout.addWidget(self.creativity_selector)

        creativity_hint = QtWidgets.QLabel(
            "<b>Mode descriptions:</b><br>"
            "🛡️ <b>Safe:</b> Literal, conservative tags<br>"
            "✨ <b>Creative:</b> Balanced, richer scenes<br>"
            "🔞 <b>Mature:</b> Adult-only, suggestive content<br>"
            "💀 <b>Extreme:</b> Strongest atmosphere & storytelling"
        )
        creativity_hint.setStyleSheet("color: #9ecbff; font-size: 10px; padding: 5px;")
        creativity_hint.setWordWrap(True)
        creativity_layout.addWidget(creativity_hint)

        desc_layout.addWidget(creativity_group)

        input_desc_group = QtWidgets.QGroupBox("✍️ Description Input")
        input_desc_layout = QtWidgets.QVBoxLayout(input_desc_group)
        input_desc_layout.setSpacing(8)

        desc_hint = QtWidgets.QLabel(
            "Describe what you want to see, and AI will generate Danbooru tags:"
        )
        desc_hint.setStyleSheet("color: #4da6ff; font-size: 12px; font-weight: bold;")
        input_desc_layout.addWidget(desc_hint)

        self.description_input = QtWidgets.QPlainTextEdit()
        self.description_input.setPlaceholderText(
            "Examples:\n"
            "• A girl with long black hair and red eyes, wearing a maid outfit\n"
            "• Anime boy with blue eyes and blonde hair, holding a sword\n"
            "• Beautiful landscape with mountains and sunset in fantasy art style\n"
            "• Character with animal ears, tail, and wearing school uniform"
        )
        self.description_input.setMinimumHeight(160)
        self.description_input.setMaximumHeight(200)
        self.description_input.setStyleSheet(
            "background-color: #0d0d0d; color: #ffffff; border: 1px solid #444; "
            "border-radius: 5px; padding: 8px;"
        )
        input_desc_layout.addWidget(self.description_input)

        self.generate_from_desc_btn = QtWidgets.QPushButton("✨ Generate Tags from Description")
        self.generate_from_desc_btn.setObjectName("tagBtn")
        self.generate_from_desc_btn.setMinimumHeight(45)
        self.generate_from_desc_btn.setToolTip(
            "AI will analyze your description and generate matching Danbooru tags"
        )
        self.generate_from_desc_btn.setStyleSheet("""
            QPushButton#tagBtn {
                background-color: #0059b3;
                font-weight: bold;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton#tagBtn:hover {
                background-color: #0073e6;
            }
        """)
        self.generate_from_desc_btn.clicked.connect(self._generate_tags_from_description)
        input_desc_layout.addWidget(self.generate_from_desc_btn)

        desc_layout.addWidget(input_desc_group)

        tags_group = QtWidgets.QGroupBox("🏷️ Generated Tags")
        tags_layout = QtWidgets.QVBoxLayout(tags_group)

        self.desc_tags_display = QtWidgets.QPlainTextEdit()
        self.desc_tags_display.setReadOnly(True)
        self.desc_tags_display.setPlaceholderText(
            "Generated Danbooru tags will appear here after processing..."
        )
        self.desc_tags_display.setStyleSheet(
            "background-color: #0d0d0d; color: #66ff66; font-family: monospace; "
            "font-size: 11px; border-radius: 5px; padding: 8px;"
        )
        self.desc_tags_display.setMinimumHeight(100)
        self.desc_tags_display.setMaximumHeight(160)
        tags_layout.addWidget(self.desc_tags_display)

        copy_tags_btn = QtWidgets.QPushButton("📋 Copy Tags to Clipboard")
        copy_tags_btn.setToolTip("Copy generated tags as comma-separated list")
        copy_tags_btn.setStyleSheet(
            "background-color: #ff9933; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px;"
        )
        copy_tags_btn.clicked.connect(self._copy_description_tags)
        self._copy_tags_btn = copy_tags_btn
        tags_layout.addWidget(copy_tags_btn)

        desc_layout.addWidget(tags_group)

        self.tabs.addTab(desc_tab, "Description Tagger")

        # ── Status bar ─────────────────────────────────────────────────────
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready. Load images to start tagging.")

        self.help_btn = QtWidgets.QPushButton("❓ Help")
        self.help_btn.setMaximumHeight(25)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3b3b3b;
                border: 1px solid #4da6ff;
            }
        """)
        self.help_btn.clicked.connect(self.show_help)
        self.statusbar.addPermanentWidget(self.help_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(250)
        self.progress.setFormat("Processing: %p%")
        self.statusbar.addPermanentWidget(self.progress)

        self._set_export_enabled(False)

        # Setup caption completer
        self._setup_caption_completer()

        # ── Drop overlay ────────────────────────────────────────────────────
        self._drop_overlay = QtWidgets.QLabel(self)
        self._drop_overlay.setText("📥\n\nDrop images here\nto upload")
        self._drop_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self._drop_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 210);
                color: #4da6ff;
                font-size: 28px;
                font-weight: bold;
                border: 3px dashed #4da6ff;
                border-radius: 20px;
                padding: 40px;
            }
        """)
        self._drop_overlay.setVisible(False)

    # ==================================================================
    # event filter (overrides QMainWindow)
    # ==================================================================

    def eventFilter(self, obj, event) -> bool:
        """Route caption-completer events before default handling."""
        if self._completer_event_filter(obj, event):
            return True
        return super().eventFilter(obj, event)

    # ==================================================================
    # Help / About
    # ==================================================================

    def show_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Img-Tagboru",
            "Img-Tagboru v1.2\n\n"
            "Local Anime Image Tagger\n"
            "WD14-style tagging for anime images and LoRA training.\n\n"
            "Features:\n"
            "• Local image tagging with WD14 models\n"
            "• Batch processing with drag & drop\n"
            "• Description-to-tags with Ollama LLM\n"
            "• Export captions for training datasets\n\n"
            "Built with Python, PyQt6, and Deep Learning.",
        )

    # ==================================================================
    # UI helpers
    # ==================================================================

    def _set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.setEnabled(enabled)
        self.export_all_btn.setEnabled(enabled)
        self.export_zip_btn.setEnabled(enabled)

    def _selected_categories(self) -> set[str]:
        selected: set[str] = set()
        if self.general_enabled.isChecked():
            selected.add("general")
        if self.character_enabled.isChecked():
            selected.add("character")
        return selected

    # ==================================================================
    # Image / path loading
    # ==================================================================

    def _open_image_safe(self, path: Path) -> Image.Image | None:
        try:
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

    def _load_paths(self, paths: Sequence[Path]) -> None:
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

        self.results = []
        self._single_results = {}
        self._active_result_index = -1
        self.table.setRowCount(0)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText("")
        self.caption_edit.blockSignals(False)
        self._set_export_enabled(False)

        self.pending_paths = valid_paths
        self.pending_label.setText(f"Loaded {len(self.pending_paths)} image(s).")
        self.tag_btn.setEnabled(bool(self.pending_paths))

        self.result_list.blockSignals(True)
        self.result_list.clear()
        for p in self.pending_paths:
            self.result_list.addItem(p.name)
        self.result_list.blockSignals(False)
        if self.pending_paths:
            self.result_list.setCurrentRow(0)
            self.show_result(0)
        elif skipped:
            self.statusbar.showMessage("No valid images found in drop/paste input.", 7000)

    def open_images(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Image",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if paths:
            self._load_paths([Path(path) for path in paths])

    # ==================================================================
    # Drag & drop
    # ==================================================================

    def _show_drop_overlay(self) -> None:
        """Position and show the drop overlay to cover the central widget."""
        if hasattr(self, '_drop_overlay'):
            rect = self.centralWidget().rect()
            self._drop_overlay.setGeometry(rect)
            self._drop_overlay.setVisible(True)
            self._drop_overlay.raise_()

    def _hide_drop_overlay(self) -> None:
        """Hide the drop overlay."""
        if hasattr(self, '_drop_overlay'):
            self._drop_overlay.setVisible(False)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if (
            mime_data.hasUrls()
            or mime_data.hasImage()
            or mime_data.hasHtml()
            or mime_data.hasText()
            or any(fmt.startswith("image/") for fmt in mime_data.formats())
            or mime_data.hasFormat("application/octet-stream")
        ):
            event.acceptProposedAction()
            self._show_drop_overlay()
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self._hide_drop_overlay()
        super().dragLeaveEvent(event)

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

            headers_variants = [
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://danbooru.donmai.us/",
                    "Accept": "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Sec-Fetch-Dest": "image",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "same-site",
                    "Cache-Control": "max-age=0",
                },
                {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
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

                        if (
                            content_type
                            and content_type not in ("application/octet-stream", "")
                            and "image/" not in content_type
                        ):
                            print(f"[DEBUG] Rejected due to Content-Type: {content_type}")
                            continue

                        try:
                            img = Image.open(io.BytesIO(data))
                            img.load()
                            print(f"[DEBUG] PIL validation successful, format: {img.format}")
                        except (UnidentifiedImageError, OSError) as e:
                            print(f"[DEBUG] PIL validation failed: {e}")
                            continue

                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        ext = (img.format or "jpg").lower() if hasattr(img, "format") else "jpg"
                        ext = ext if ext in ("png", "jpeg", "jpg", "webp", "bmp", "gif") else "jpg"
                        temp_path = temp_dir / f"web_{uuid4().hex[:8]}.{ext}"
                        temp_path.write_bytes(data)
                        print(f"[DEBUG] Saved to {temp_path}")
                        return temp_path
                except urllib.error.HTTPError as e:
                    print(f"[DEBUG] HTTP Error {e.code} on attempt {attempt + 1}")
                    if attempt == len(headers_variants) - 1:
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
            base_match = re.search(
                r'<base[^>]*href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE
            )
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

        unique: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        self._hide_drop_overlay()
        mime_data = event.mimeData()

        # 1) Local files/folders
        if mime_data.hasUrls():
            urls = mime_data.urls()
            local_paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
            if local_paths:
                files: list[Path] = []
                for path in local_paths:
                    if path.is_dir():
                        files.extend(
                            [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
                        )
                    else:
                        files.append(path)
                if files:
                    self._load_paths(files)
                    self.statusbar.showMessage(f"Loaded {len(files)} dropped file(s).", 5000)
                    event.acceptProposedAction()
                    return

        # 2) Qt native image
        if mime_data.hasImage():
            print("[DEBUG] Attempting to use Qt's native image support via hasImage()")
            try:
                image = QtGui.QImage(mime_data.imageData())
                if not image.isNull():
                    print("[DEBUG] Successfully got QImage from mime_data")
                    temp_path = self._save_qimage_temp(image, "dragged")
                    if temp_path is not None:
                        print(f"[DEBUG] Saved QImage to {temp_path}")
                        self._load_paths([temp_path])
                        self.statusbar.showMessage("Loaded dropped image.", 5000)
                        event.acceptProposedAction()
                        return
            except Exception as e:
                print(f"[DEBUG] Qt image handling failed: {e}")

        # 3) Raw image data from any MIME type
        print(f"[DEBUG] Available MIME formats: {mime_data.formats()}")
        for fmt in mime_data.formats():
            print(f"[DEBUG] Trying to extract image from format: {fmt}")
            try:
                image_bytes = mime_data.data(fmt)
                if image_bytes and len(image_bytes) > 500:
                    print(f"[DEBUG] Found {len(image_bytes)} bytes in format {fmt}")

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

                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        img.load()
                        print(f"[DEBUG] Successfully parsed as {img.format}")

                        temp_dir = Path(tempfile.gettempdir()) / "img-tagger-web"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        ext = (img.format or "jpg").lower() if hasattr(img, "format") else "jpg"
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

        # 4) Web URLs
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

        all_formats = mime_data.formats()
        print(f"[DEBUG] Drop not recognized. Available MIME types: {all_formats}")
        self.statusbar.showMessage(
            "Drop not recognized. Try: drag image file, drag from website, or Ctrl+V.",
            7000,
        )
        super().dropEvent(event)

    # ==================================================================
    # Keyboard / paste
    # ==================================================================

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            clipboard = QtWidgets.QApplication.clipboard()
            mime_data = clipboard.mimeData()

            if mime_data.hasUrls():
                paths = [Path(url.toLocalFile()) for url in mime_data.urls() if url.isLocalFile()]
                if paths:
                    self._load_paths(paths)
                    self.statusbar.showMessage("Loaded pasted file path(s).", 5000)
                    event.accept()
                    return

            for url in self._extract_web_image_candidates(mime_data):
                downloaded = self._download_web_image_to_temp(url)
                if downloaded is not None:
                    self._load_paths([downloaded])
                    self.statusbar.showMessage("Loaded pasted web image URL.", 5000)
                    event.accept()
                    return

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

    # ==================================================================
    # Folder picker
    # ==================================================================

    def open_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open folder", str(Path.cwd()))
        if not folder:
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Include Subfolders?")
        msg_box.setText("Do you want to include images from subfolders?")
        msg_box.setInformativeText(f"Selected folder:\n{folder}")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Cancel:
            return

        root = Path(folder)
        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            paths = [
                path
                for path in sorted(root.rglob("*"))
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            paths = [
                path
                for path in sorted(root.glob("*"))
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]

        self._load_paths(paths)

    # ==================================================================
    # Tagging (core)
    # ==================================================================

    def _tag_image(self, image: Image.Image) -> list:
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
        self.table.blockSignals(True)
        self.result_list.blockSignals(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusbar.showMessage("Tagging images...")

        blacklist = split_tags(self.blacklist.toPlainText())
        whitelist = split_tags(self.whitelist.toPlainText())
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
                frame = frame_from_predictions(predictions)
                if not frame.empty:
                    if allowed_categories:
                        frame = frame[frame["category"].isin(allowed_categories)].copy()
                    frame = apply_filters(frame, blacklist, whitelist)
                    frame = sort_frame(frame, self.sort_mode.currentText())
                caption = frame_to_caption(frame, include_scores=self.include_scores.isChecked())
                self.results.append(
                    TaggingResult(name=path.name, path=path, image=image, frame=frame, caption=caption)
                )
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

    def _update_tag_selected_button(self) -> None:
        index = self.result_list.currentRow()
        has_pending = 0 <= index < len(self.pending_paths)
        has_result = 0 <= index < len(self.results)
        self.tag_selected_btn.setEnabled(has_pending or has_result)

    def process_single_image(self) -> None:
        index = self.result_list.currentRow()

        if index < 0:
            self.statusbar.showMessage("No image selected.", 3000)
            return

        if 0 <= index < len(self.results):
            existing = self.results[index]
            path = existing.path
            image = existing.image
        elif index in self._single_results:
            existing = self._single_results[index]
            path = existing.path
            image = existing.image
        elif 0 <= index < len(self.pending_paths):
            path = self.pending_paths[index]
            image = self._open_image_safe(path)
            if image is None:
                self.statusbar.showMessage(f"Cannot load image: {path.name}", 3000)
                return
        else:
            self.statusbar.showMessage("No image selected.", 3000)
            return

        self.statusbar.showMessage(f"Tagging {path.name}...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            predictions = self._tag_image(image)
            frame = frame_from_predictions(predictions)
            if not frame.empty:
                allowed_categories = self._selected_categories()
                if allowed_categories:
                    frame = frame[frame["category"].isin(allowed_categories)].copy()
                blacklist = split_tags(self.blacklist.toPlainText())
                whitelist = split_tags(self.whitelist.toPlainText())
                frame = apply_filters(frame, blacklist, whitelist)
                frame = sort_frame(frame, self.sort_mode.currentText())
            else:
                frame = pd.DataFrame(
                    columns=["include", "rank", "tag", "confidence", "category"]
                )
            caption = frame_to_caption(frame, include_scores=self.include_scores.isChecked())

            new_result = TaggingResult(
                name=path.name, path=path, image=image, frame=frame, caption=caption
            )
            if 0 <= index < len(self.results):
                self.results[index] = new_result
            else:
                self._single_results[index] = new_result

            self._active_result_index = index
            self._frame_to_table(frame)
            self.caption_edit.blockSignals(True)
            self.caption_edit.setPlainText(caption)
            self.caption_edit.blockSignals(False)
            self._set_export_enabled(True)

            tag_count = len(frame) if not frame.empty else 0
            self.statusbar.showMessage(f"Tagged {path.name} ({tag_count} tags)")
        except Exception as e:
            self.statusbar.showMessage(f"Error tagging: {str(e)}", 5000)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ==================================================================
    # Result selection / navigation
    # ==================================================================

    def _current_index(self) -> int:
        return self.result_list.currentRow()

    def _current_result(self) -> TaggingResult | None:
        index = self._current_index()
        if 0 <= index < len(self.results):
            return self.results[index]
        if index in self._single_results:
            return self._single_results[index]
        return None

    def _sync_current_result(self) -> None:
        result = self._current_result()
        if result is None:
            return
        result.frame = self._table_to_frame()
        result.caption = self.caption_edit.toPlainText().strip()

    def show_result(self, index: int) -> None:
        if not (0 <= index < len(self.results)):
            if index in self._single_results:
                if 0 <= self._active_result_index < len(self.results) and self.table.rowCount() > 0:
                    prev = self.results[self._active_result_index]
                    prev.frame = self._table_to_frame()
                    prev.caption = self.caption_edit.toPlainText().strip()
                elif self._active_result_index in self._single_results and self.table.rowCount() > 0:
                    prev = self._single_results[self._active_result_index]
                    prev.frame = self._table_to_frame()
                    prev.caption = self.caption_edit.toPlainText().strip()
                result = self._single_results[index]
                self._active_result_index = index
                self._set_image(result.image)
                self._frame_to_table(result.frame)
                self.caption_edit.blockSignals(True)
                self.caption_edit.setPlainText(result.caption)
                self.caption_edit.blockSignals(False)
                return
            if 0 <= index < len(self.pending_paths):
                self._active_result_index = -1
                img = self._open_image_safe(self.pending_paths[index])
                if img is not None:
                    self._set_image(img)
                else:
                    self.image_label.clear()
                    self.statusbar.showMessage("Could not preview this image file.", 7000)
                self.table.setRowCount(0)
                self.caption_edit.blockSignals(True)
                self.caption_edit.setPlainText("")
                self.caption_edit.blockSignals(False)
            return

        if 0 <= self._active_result_index < len(self.results) and self.table.rowCount() > 0:
            prev = self.results[self._active_result_index]
            prev.frame = self._table_to_frame()
            prev.caption = self.caption_edit.toPlainText().strip()

        result = self.results[index]
        self._active_result_index = index
        self._set_image(result.image)
        self._frame_to_table(result.frame)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(result.caption)
        self.caption_edit.blockSignals(False)

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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._preview_image is not None:
            self._set_image(self._preview_image)
        # Keep the drop overlay sized to the central widget
        if hasattr(self, '_drop_overlay') and self._drop_overlay.isVisible():
            rect = self.centralWidget().rect()
            self._drop_overlay.setGeometry(rect)

    # ==================================================================
    # Table ↔ frame ↔ caption sync
    # ==================================================================

    def _frame_to_table(self, frame: pd.DataFrame) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row_index, row in enumerate(frame.itertuples(index=False)):
            self.table.insertRow(row_index)

            include_item = QtWidgets.QTableWidgetItem()
            include_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            include_item.setCheckState(
                QtCore.Qt.Checked if bool(row.include) else QtCore.Qt.Unchecked
            )
            self.table.setItem(row_index, 0, include_item)

            rank_item = QtWidgets.QTableWidgetItem(str(int(row.rank)))
            rank_item.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
            )
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
        rows: list[dict] = []
        for row_index in range(self.table.rowCount()):
            include_item = self.table.item(row_index, 0)
            rows.append(
                {
                    "include": (
                        include_item.checkState() == QtCore.Qt.Checked
                        if include_item
                        else False
                    ),
                    "rank": (
                        int(rank_item.text())
                        if (rank_item := self.table.item(row_index, 1))
                        and rank_item.text().strip().isdigit()
                        else row_index + 1
                    ),
                    "tag": (
                        (tag_item := self.table.item(row_index, 2)).text()
                        if (tag_item := self.table.item(row_index, 2))
                        else ""
                    ),
                    "confidence": (
                        float(confidence_item.text())
                        if (confidence_item := self.table.item(row_index, 3))
                        else 0.0
                    ),
                    "category": (
                        (category_item := self.table.item(row_index, 4)).text()
                        if (category_item := self.table.item(row_index, 4))
                        else ""
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = sort_frame(frame, self.sort_mode.currentText())
        return frame

    def on_table_changed(self, *_args) -> None:
        result = self._current_result()
        if result is None:
            return
        result.frame = self._table_to_frame()
        result.caption = frame_to_caption(result.frame, include_scores=self.include_scores.isChecked())
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(result.caption)
        self.caption_edit.blockSignals(False)

    def apply_caption_text(self) -> None:
        result = self._current_result()
        if result is None:
            return
        tags = split_tags(self.caption_edit.toPlainText())
        tag_order = {tag.lower(): index + 1 for index, tag in enumerate(tags)}
        frame = result.frame.copy()
        lower_tags = {tag.lower() for tag in tags}
        frame["include"] = frame["tag"].astype(str).str.lower().isin(lower_tags)
        frame["rank"] = [tag_order.get(str(tag).lower(), 9999) for tag in frame["tag"]]
        frame = sort_frame(frame, self.sort_mode.currentText())
        result.frame = frame
        result.caption = frame_to_caption(frame, include_scores=self.include_scores.isChecked())
        self._frame_to_table(frame)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(result.caption)
        self.caption_edit.blockSignals(False)

    # ==================================================================
    # Description tagger (Tab 2)
    # ==================================================================

    def _refresh_available_models(self) -> None:
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
                    idx = self.model_selector.findData(preferred_model)
                    if idx >= 0:
                        self.model_selector.setCurrentIndex(idx)
        except Exception:
            self.model_selector.addItem("(error loading)", None)
        finally:
            self.model_selector.blockSignals(False)

    def _generate_tags_from_description(self) -> None:
        description = self.description_input.toPlainText().strip()
        if not description:
            self.statusbar.showMessage("Description is empty. Please enter a description.", 5000)
            return

        selected_model = self.model_selector.currentData()
        if not selected_model:
            self.desc_tags_display.setPlainText(
                "⚠️ Error:\n\nNo model selected. Refresh models or start Ollama."
            )
            self.statusbar.showMessage("Tag generation failed.", 5000)
            return

        selected_creativity = self.creativity_selector.currentData() or "creative"
        self._last_creativity_mode = selected_creativity

        self.generate_from_desc_btn.setEnabled(False)
        self.desc_tags_display.setPlainText(
            "⏳ Generating tags... (this may take a while)\n\n"
            f"Mode: {selected_creativity.capitalize()}\n"
            "⚠️ TIP: Enable GPU in Ollama for faster generation!"
        )
        self.statusbar.showMessage(
            f"Connecting to Ollama and generating tags in {selected_creativity} mode..."
        )

        self._tag_worker = DescriptionTagWorker(description, selected_model, selected_creativity)
        self._tag_worker.finished.connect(self._on_tags_generated)
        self._tag_worker.error.connect(self._on_tag_generation_error)
        self._tag_worker.start()

    def _on_tags_generated(self, result: DescriptionTagResult) -> None:
        self._last_description_tags = result.tags

        tags_output = ", ".join(result.tags)

        if len(result.tags) == 0:
            display_text = (
                "⚠️ No output generated\n\n"
                "Try refining your description with more visual details."
            )
            self._copy_tags_btn.setEnabled(False)
        else:
            display_text = (
                f"✓ Generated prompt ({len(result.tags)} terms) "
                f"[{self._last_creativity_mode.capitalize()} mode]:\n\n{tags_output}"
            )
            self._copy_tags_btn.setEnabled(True)

        self.desc_tags_display.setPlainText(display_text)
        self.statusbar.showMessage(
            f"✓ Generated {len(result.tags)} prompt terms in {self._last_creativity_mode} mode.",
            5000,
        )
        self.generate_from_desc_btn.setEnabled(True)
        self._tag_worker = None

    def _on_tag_generation_error(self, error_msg: str) -> None:
        self.desc_tags_display.setPlainText(f"⚠️ Error:\n\n{error_msg}")
        self.statusbar.showMessage("Tag generation failed.", 5000)
        self.generate_from_desc_btn.setEnabled(True)
        self._copy_tags_btn.setEnabled(False)
        self._tag_worker = None

    def _copy_description_tags(self) -> None:
        if not hasattr(self, '_last_description_tags') or not self._last_description_tags:
            self.statusbar.showMessage("No tags to copy.", 2000)
            return
        tags_text = ", ".join(self._last_description_tags)
        QtWidgets.QApplication.clipboard().setText(tags_text)
        self.statusbar.showMessage(
            f"✓ Copied {len(self._last_description_tags)} tags to clipboard", 3000
        )

    # ==================================================================
    # Export
    # ==================================================================

    def export_caption(self) -> None:
        result = self._current_result()
        if result is None or result.path is None:
            return
        self._sync_current_result()
        default_path = str(result.path.with_suffix(".txt"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save caption", default_path, "Text (*.txt)"
        )
        if not path:
            return
        Path(path).write_text(result.caption, encoding="utf-8")
        self.statusbar.showMessage(f"Saved caption to {path}")

    def export_all_captions(self) -> None:
        if not self.results:
            return
        self._sync_current_result()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output folder", str(Path.cwd())
        )
        if not folder:
            return
        output = Path(folder)
        for result in self.results:
            if result.path is None:
                continue
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
        Path(path).write_bytes(export_zip_from_results(self.results))
        self.statusbar.showMessage(f"Saved zip to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QtWidgets.QApplication([])
    app.setApplicationName("Img-Tagboru")
    app.setApplicationDisplayName("Img-Tagboru")
    app.setApplicationVersion("1.2")

    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
