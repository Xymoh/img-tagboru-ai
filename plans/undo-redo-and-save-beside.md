# Plan: Undo/Redo & Save Beside Source Images

## Feature 1: Undo/Redo for Caption Edits & Tag Toggles

### Overview
Track mutations to the active `TaggingResult` (frame + caption) through a pair of stacks, expose via Ctrl+Z / Ctrl+Y keyboard shortcuts and optional toolbar buttons.

### State to Track
Each undo entry stores a deep-copy of the active result's `frame` (DataFrame) and `caption` (str).

### Mutation Points (where to push undo)
| Trigger | Method | Notes |
|---------|--------|-------|
| Checkbox toggled (Include column) | `on_table_changed` | Qt auto-toggles → `itemChanged` → `on_table_changed` runs AFTER change. Snapshot BEFORE modifying result in `on_table_changed`. |
| Rank cell edited | `on_table_changed` | Same path as checkbox — captured by the same pre-snapshot. |
| Apply Caption button | `apply_caption_text` | Manual sync from caption text → table. Snapshot before modifying. |

### Implementation
```python
# In __init__:
self._undo_stack: list[tuple[pd.DataFrame, str]] = []
self._redo_stack: list[tuple[pd.DataFrame, str]] = []
self._undo_action = QtGui.QAction("Undo", self)
self._undo_action.setShortcut(QtGui.QKeySequence.Undo)
self._undo_action.triggered.connect(self._undo)
self.addAction(self._undo_action)
self._redo_action = QtGui.QAction("Redo", self)
self._redo_action.setShortcut(QtGui.QKeySequence("Ctrl+Y"))
self._redo_action.triggered.connect(self._redo)
self.addAction(self._redo_action)

# Snapshot helper:
def _push_undo_state(self):
    r = self._current_result()
    if r is None:
        return
    state = (r.frame.copy(), r.caption)
    # Skip if identical to top of stack (debounce)
    if self._undo_stack and self._undo_stack[-1][1] == state[1]:
        return
    self._undo_stack.append(state)
    self._redo_stack.clear()

# In on_table_changed – add at TOP before modifications:
self._push_undo_state()

# In apply_caption_text – add at TOP before modifications:
self._push_undo_state()

# Undo:
def _undo(self):
    if not self._undo_stack:
        return
    r = self._current_result()
    if r:
        self._redo_stack.append((r.frame.copy(), r.caption))
    frame, caption = self._undo_stack.pop()
    if r:
        r.frame = frame
        r.caption = caption
        self._frame_to_table(frame)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(caption)
        self.caption_edit.blockSignals(False)

# Redo:
def _redo(self):
    if not self._redo_stack:
        return
    r = self._current_result()
    if r:
        self._undo_stack.append((r.frame.copy(), r.caption))
    frame, caption = self._redo_stack.pop()
    if r:
        r.frame = frame
        r.caption = caption
        self._frame_to_table(frame)
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(caption)
        self.caption_edit.blockSignals(False)
```

### Files Changed
- [`main_window.py`](frontend/native/main_window.py) — add stacks, actions, `_push_undo_state`, `_undo`, `_redo`, modify `on_table_changed` and `apply_caption_text`

---

## Feature 2: Save .txt Beside Source Images

### Overview
Add a new export button "💾 Save Beside Source" that saves all captions as .txt files in the same folder as each source image, using the same filename (e.g., `photo.png` → `photo.txt`). No folder picker — it writes directly.

### Implementation
```python
# New button in caption_buttons row:
self.export_beside_btn = QtWidgets.QPushButton("💾 Save Beside Source")
self.export_beside_btn.setToolTip("Save all captions as .txt files next to their source images")
self.export_beside_btn.clicked.connect(self.export_beside_source)
caption_buttons.addWidget(self.export_beside_btn)

# Handler:
def export_beside_source(self) -> None:
    if not self.results:
        return
    self._sync_current_result()
    saved = 0
    for r in self.results:
        if r.path is None:
            continue
        txt_path = r.path.with_suffix(".txt")
        try:
            txt_path.write_text(r.caption, encoding="utf-8")
            saved += 1
        except OSError as e:
            self.statusbar.showMessage(f"Error saving {txt_path.name}: {e}", 5000)
    self.statusbar.showMessage(f"Saved {saved} caption(s) beside source images.")
```

### Files Changed
- [`main_window.py`](frontend/native/main_window.py) — add button, handler, and `_set_export_enabled` update

### Button Layout (updated)
| # | Button | Action |
|---|--------|--------|
| 1 | 🔄 Apply Caption | Sync caption text → table |
| 2 | 💾 Save Current | Save single caption with dialog |
| 3 | 💾 Save All | Save all to chosen folder |
| 4 | 💾 Save Beside Source | **NEW** — Save all beside images, no dialog |
| 5 | 📦 Export ZIP | Download all as ZIP |
