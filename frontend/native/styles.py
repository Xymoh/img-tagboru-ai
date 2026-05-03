from __future__ import annotations

import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Checkmark SVG
# ---------------------------------------------------------------------------

def _write_checkmark_svg() -> str:
    """Write a white checkmark SVG to a temp file and return its path."""
    svg_content = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
        '<polyline points="2,7 6,11 12,3" stroke="white" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        '</svg>'
    )
    tmp_dir = Path(tempfile.gettempdir()) / "img-tagger-assets"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    svg_path = tmp_dir / "checkmark.svg"
    svg_path.write_text(svg_content, encoding="utf-8")
    return str(svg_path).replace("\\", "/")


CHECKMARK_SVG_PATH = _write_checkmark_svg()


# ---------------------------------------------------------------------------
# Application stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """Return the complete dark-theme Qt stylesheet."""

    return """
        QMainWindow {
            background-color: #0d0d0d;
        }
        QWidget {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #333;
            margin-top: 15px;
            padding-top: 20px;
            border-radius: 8px;
            background-color: #1f1f1f;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #4da6ff;
            font-size: 13px;
        }
        QPushButton {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #444;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #3b3b3b;
            border: 1px solid #4da6ff;
        }
        QPushButton:pressed {
            background-color: #1a1a1a;
        }
        QPushButton#tagBtn {
            background-color: #0059b3;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
        }
        QPushButton#tagBtn:hover {
            background-color: #0073e6;
        }
        QPushButton#tagSelectedBtn {
            background-color: #1a5c1a;
            color: #aaffaa;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 16px;
            border: 1px solid #2d8a2d;
        }
        QPushButton#tagSelectedBtn:hover {
            background-color: #236b23;
            border: 1px solid #44bb44;
            color: #ccffcc;
        }
        QPushButton#tagSelectedBtn:pressed {
            background-color: #0f3d0f;
        }
        QPushButton#tagSelectedBtn:disabled {
            background-color: #1a2e1a;
            color: #557755;
            border: 1px solid #2a3d2a;
        }
        QLabel#alertLabel {
            color: #ff9933;
            font-weight: bold;
        }
        QLabel {
            color: #e0e0e0;
        }
        QTableWidget {
            background-color: #0d0d0d;
            alternate-background-color: #1a1a1a;
            gridline-color: #333;
            border: 1px solid #444;
            border-radius: 5px;
            selection-background-color: #0059b3;
        }
        QTableWidget::item {
            padding: 4px;
            border-bottom: 1px solid #222;
        }
        QTableWidget::item:selected {
            background-color: #0059b3;
            color: white;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #4da6ff;
            padding: 6px;
            border: 1px solid #444;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            border-radius: 5px;
            background-color: #1a1a1a;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #1a1a1a;
            color: #4da6ff;
            font-weight: bold;
            border-bottom: 2px solid #4da6ff;
        }
        QTabBar::tab:hover {
            background-color: #3b3b3b;
        }
        QProgressBar {
            border: 1px solid #444;
            border-radius: 3px;
            background-color: #2b2b2b;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background-color: #0059b3;
            border-radius: 2px;
        }
        QStatusBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-top: 1px solid #444;
        }
        QListWidget {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 5px;
        }
        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #222;
        }
        QListWidget::item:selected {
            background-color: #0059b3;
            color: white;
            border-radius: 3px;
        }
        QPlainTextEdit {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 5px;
            color: #e0e0e0;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #0d0d0d;
            border: 1px solid #444;
            border-radius: 3px;
            padding: 4px;
            color: #e0e0e0;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 8px solid #4da6ff;
            width: 0px;
            height: 0px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #444;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
        }
        QCheckBox {
            color: #e0e0e0;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #666;
            border-radius: 4px;
            background-color: #0d0d0d;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #4da6ff;
            background-color: #1a1a1a;
        }
        QCheckBox::indicator:checked {
            background-color: #0059b3;
            border: 2px solid #4da6ff;
            image: url(CHECKMARK_PATH);
        }
        QCheckBox::indicator:checked:hover {
            background-color: #0073e6;
            border: 2px solid #66b3ff;
            image: url(CHECKMARK_PATH);
        }
        QCheckBox::indicator:unchecked:hover {
            background-color: #1a1a1a;
        }
    """.replace("CHECKMARK_PATH", CHECKMARK_SVG_PATH)
