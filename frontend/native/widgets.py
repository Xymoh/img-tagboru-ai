from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Custom delegate for table checkbox column with styled checkmark
# ---------------------------------------------------------------------------

class CheckboxDelegate(QtWidgets.QStyledItemDelegate):
    """Draws a styled checkbox with a visible checkmark in the table."""

    def paint(self, painter, option, index):
        painter.save()
        checked = index.data(QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked

        # Draw cell background
        if option.state & QtWidgets.QStyle.State_Selected:
            painter.fillRect(option.rect, QtGui.QColor("#0059b3"))
        else:
            bg = QtGui.QColor("#0d0d0d") if index.row() % 2 == 0 else QtGui.QColor("#1a1a1a")
            painter.fillRect(option.rect, bg)

        # Box dimensions
        box_size = 16
        cx = option.rect.center().x()
        cy = option.rect.center().y()
        box_rect = QtCore.QRect(cx - box_size // 2, cy - box_size // 2, box_size, box_size)

        if checked:
            # Filled blue box
            painter.setBrush(QtGui.QColor("#0059b3"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#4da6ff"), 1.5))
            painter.drawRoundedRect(box_rect, 3, 3)
            # Draw white checkmark
            pen = QtGui.QPen(
                QtGui.QColor("#ffffff"), 2.2,
                QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            x, y = box_rect.x(), box_rect.y()
            w, h = box_rect.width(), box_rect.height()
            p1 = QtCore.QPointF(x + w * 0.15, y + h * 0.50)
            p2 = QtCore.QPointF(x + w * 0.42, y + h * 0.75)
            p3 = QtCore.QPointF(x + w * 0.85, y + h * 0.22)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
        else:
            # Empty dark box
            painter.setBrush(QtGui.QColor("#0d0d0d"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#666666"), 1.5))
            painter.drawRoundedRect(box_rect, 3, 3)

        painter.restore()

    def sizeHint(self, option, index):
        return QtCore.QSize(40, 26)


# ---------------------------------------------------------------------------
# Help dialog
# ---------------------------------------------------------------------------

class HelpDialog(QtWidgets.QDialog):
    """Help dialog with usage instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📖 User Guide - Img-Tagboru")
        self.resize(700, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QTextBrowser {
                background-color: #0d0d0d;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 10px;
                color: #ffffff;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("🏷️ Img-Tagboru - User Guide")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4da6ff; margin-bottom: 10px;")
        layout.addWidget(title)

        # Content browser
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml("""
        <h2 style="color: #4da6ff;">Quick Start Guide</h2>

        <h3 style="color: #66ff66;">📁 Loading Images</h3>
        <ul>
            <li><b>Drag & Drop:</b> Drag image files or folders directly onto the app</li>
            <li><b>Clipboard:</b> Copy an image and press Ctrl+V</li>
            <li><b>Buttons:</b> Use "Open Image" or "Open Folder" buttons</li>
            <li><b>Web Images:</b> Copy image URL and paste with Ctrl+V</li>
        </ul>

        <h3 style="color: #66ff66;">⚙️ Tagging Settings</h3>
        <ul>
            <li><b>General Threshold (0.25-0.40):</b> Lower = more tags, may include false positives</li>
            <li><b>Character Threshold (0.80-0.95):</b> Higher = only confident character matches</li>
            <li><b>Max Tags:</b> Limit tags per image (40-80 is typical for training)</li>
            <li><b>MCut:</b> Automatic threshold detection (overrides manual settings)</li>
        </ul>

        <h3 style="color: #66ff66;">🏷️ Working with Tags</h3>
        <ul>
            <li><b>Include Column:</b> Uncheck to exclude tags from caption</li>
            <li><b>Rank:</b> Manually order tags (lower = appears first)</li>
            <li><b>Blacklist:</b> Tags to always exclude (e.g., "blurry, lowres")</li>
            <li><b>Whitelist:</b> Only include these tags if specified</li>
            <li><b>Sort Mode:</b> Change how tags are displayed/ordered</li>
        </ul>

        <h3 style="color: #66ff66;">💾 Exporting Results</h3>
        <ul>
            <li><b>Save Current TXT:</b> Save caption for selected image</li>
            <li><b>Save All TXT:</b> Save all captions to a folder</li>
            <li><b>Export ZIP:</b> Download all captions in a ZIP file</li>
            <li><b>Caption Format:</b> "tag1, tag2, tag3" ready for training</li>
        </ul>

        <h3 style="color: #66ff66;">📝 Description Tagger (Tab 2)</h3>
        <ul>
            <li>Describe what you want to see in English</li>
            <li>AI generates Danbooru-style tags from your description</li>
            <li>Requires Ollama installed (see setup below)</li>
            <li>Choose creativity mode: Safe → Creative → Mature</li>
            <li><b>Re-run for variety:</b> The AI uses temperature sampling — running the same prompt again can produce different (sometimes better) results</li>
        </ul>

        <h3 style="color: #ff9933;">✍️ Writing Better Descriptions</h3>
        <p>The more concrete visual detail you provide, the better the tags. The AI maps descriptions to tags across these dimensions:</p>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Dimension</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Good Example</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Poor Example</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Subject</b><br><span style="color: #888;">who is in the scene?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"a knight", "two elves", "a catgirl"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">"someone", "a character"</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444;"><b>Action</b><br><span style="color: #888;">what are they doing?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"baking cookies", "kissing", "standing"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">"existing", "being"</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Setting</b><br><span style="color: #888;">where does this happen?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"in a forest clearing", "on the train"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no setting mentioned)</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444;"><b>Atmosphere</b><br><span style="color: #888;">mood / lighting</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"cozy", "stormy night", "romantic"</td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no mood mentioned)</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444;"><b>Clothing</b><br><span style="color: #888;">what are they wearing?</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">"a maid outfit", "armor and cape"<br><span style="color: #888;">(or implied by archetype: witch → hat)</span></td>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">(no clothing clues)</td>
            </tr>
        </table>

        <h4 style="color: #ffcc66;">Mode Selection Guide</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Mode</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Best For</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Tags Include</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;">🟢 Safe</td>
                <td style="padding: 6px; border: 1px solid #444;">SFW, character portraits, scenery</td>
                <td style="padding: 6px; border: 1px solid #444;">No sexual content — groping/kissing gets softened to blush/romance</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444; color: #ff9933;">🟡 Creative</td>
                <td style="padding: 6px; border: 1px solid #444;">Action, atmosphere, mild romance</td>
                <td style="padding: 6px; border: 1px solid #444;">Style/lighting tags, kissing, hand-holding, suggestive — no explicit</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #ff6666;">🔴 Mature</td>
                <td style="padding: 6px; border: 1px solid #444;">NSFW, explicit sexual content</td>
                <td style="padding: 6px; border: 1px solid #444;">Full sexual vocabulary, body language, intimate settings</td>
            </tr>
        </table>

        <h4 style="color: #ff6666;">Common Pitfalls</h4>
        <ul style="font-size: 12px;">
            <li><b>Too vague:</b> "a witch" → few generic tags. Try "a witch flying through a dark storm"</li>
            <li><b>Abstract concepts:</b> "a feeling of dread" → can't map to visual Danbooru tags. Describe what that looks like instead</li>
            <li><b>Franchise names:</b> "Hollow Knight", "Disgaea" → may collide with tag namespace. Use generic descriptions</li>
        </ul>

        <h4 style="color: #66ff66;">If Results Are Poor</h4>
        <ol style="font-size: 12px; margin: 5px 0;">
            <li><b>Add detail:</b> Make sure you have a subject + action + setting</li>
            <li><b>Re-run:</b> Temperature sampling means different runs produce different results</li>
            <li><b>Try another mode:</b> Creative often produces richer atmospheric tags than Safe</li>
            <li><b>Be explicit:</b> If a tag is missing, name the element directly</li>
        </ol>

        <h3 style="color: #ff9933;">💡 Pro Tips</h3>
        <ul>
            <li>Use blacklist to filter out unwanted tags permanently</li>
            <li>Lower general threshold catches more details but may include noise</li>
            <li>Edit tags in the table and caption updates automatically</li>
            <li>Drag multiple images or entire folders at once</li>
            <li>For LoRA training, keep captions clean with 40-80 quality tags</li>
        </ul>

        <h3 style="color: #ff66a3;">🔧 System Requirements</h3>
        <ul>
            <li><b>RAM:</b> 8GB minimum, 16GB recommended (32GB+ for Description Tagger)</li>
            <li><b>GPU:</b> Optional but speeds up tagging significantly (NVIDIA/AMD)</li>
            <li><b>Disk:</b> 5GB for models and temp files (15-25GB for LLM models)</li>
        </ul>

        <h3 style="color: #ff9933;">🤖 Description Tagger Setup</h3>
        <ol style="margin: 5px 0;">
            <li>Install <b>Ollama</b> from <a href='https://ollama.ai' style='color: #4da6ff;'>ollama.ai</a></li>
            <li>For GPU: Install NVIDIA CUDA or AMD ROCm drivers</li>
            <li>Start Ollama: run <code>ollama serve</code> in terminal</li>
            <li>Verify GPU: run <code>ollama ps</code> (shows 'GPU loaded')</li>
            <li>Pull the recommended model: <code>ollama pull richardyoung/qwen3-14b-abliterated</code></li>
        </ol>
        
        <h4 style="color: #66ff66;">⭐ Recommended Model — Tested & Verified</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Size</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Quality</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Pull Command</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #66ff66;"><b>🏆 Qwen3-14B-Abliterated</b><br><span style="color: #888;">Default · 100% reliable in testing · 16GB VRAM</span></td>
                <td style="padding: 6px; border: 1px solid #444;">14B<br><span style="color: #888;">~9GB Q4_K_M</span></td>
                <td style="padding: 6px; border: 1px solid #444;">⭐⭐⭐⭐⭐<br>Verified</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull richardyoung/qwen3-14b-abliterated</td>
            </tr>
        </table>

        <h4 style="color: #888;">Alternative Models (not tested — may produce worse or zero results)</h4>
        <table style="width:100%; border-collapse: collapse; margin: 8px 0; font-size: 11px;">
            <tr style="background-color: #2a2a2a;">
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Model</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Size</th>
                <th style="padding: 6px; text-align: left; border: 1px solid #444; color: #4da6ff;">Pull Command</th>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Qwen3.6-35B-uncensored</b><br><span style="color: #888;">MoE: 3B active / 35B total</span></td>
                <td style="padding: 6px; border: 1px solid #444;">~6GB</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull llmfan46/qwen3.6-35b-a3b-uncensored-heretic</td>
            </tr>
            <tr style="background-color: #222;">
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Gemma-4-26B-uncensored</b><br><span style="color: #888;">MoE: 4B active / 26B total</span></td>
                <td style="padding: 6px; border: 1px solid #444;">~8GB</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull llmfan46/gemma-4-26b-a4b-it-ultra-uncensored-heretic</td>
            </tr>
            <tr>
                <td style="padding: 6px; border: 1px solid #444; color: #aaa;"><b>Qwen3.5-35B-uncensored</b><br><span style="color: #888;">MoE: 3B active / 35B total</span></td>
                <td style="padding: 6px; border: 1px solid #444;">~6GB</td>
                <td style="padding: 6px; border: 1px solid #444; font-family: monospace; font-size: 10px;">ollama pull llmfan46/qwen3.5-35b-a3b-uncensored-heretic</td>
            </tr>
        </table>
        
        <p style='color: #ffcc66; font-size: 11px; margin-top: 8px;'>
            💡 <b>Tip:</b> Qwen3-14B-Abliterated is the only model verified with 100% reliability (18/18 batch test).<br>
            💡 <b>VRAM:</b> Qwen3-14B uses ~9GB at Q4_K_M, fits comfortably in 16GB GPUs.<br>
            💡 <b>Explicit content:</b> All recommended models are abliterated/uncensored for NSFW tags.
        </p>

        <hr style="border: 1px solid #444;">
        <p style="color: #9ecbff; font-size: 11px;">
            Need help? Check the
            <a href="https://github.com/your-repo/img-tagger" style="color: #4da6ff;">GitHub repository</a>
            or review the README.md file.
        </p>
        """)
        layout.addWidget(browser)

        # Close button
        close_btn = QtWidgets.QPushButton("✓ Got it!")
        close_btn.setMinimumHeight(35)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0059b3;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0073e6;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
