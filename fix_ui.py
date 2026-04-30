#!/usr/bin/env python3
"""Fix the UI to add model selection and startup loading."""

import re

with open('frontend/native_app.py', 'r') as f:
    content = f.read()

# 1. Find the line after "self._set_export_enabled(False)" and add model loading
insert_after = "        self._set_export_enabled(False)"
model_loading = "\n        # Load available models on startup\n        QtCore.QTimer.singleShot(500, self._refresh_available_models)"
if insert_after in content:
    content = content.replace(insert_after, insert_after + model_loading)
    print("✓ Added model loading at startup")

# 2. Replace the _generate_tags_from_description method
old_method = r'''    def _generate_tags_from_description\(self\) -> None:
        """Generate Danbooru tags from text description using Ollama\."""
        description = self\.description_input\.toPlainText\(\)\.strip\(\)
        if not description:
            self\.statusbar\.showMessage\("Description is empty\. Please enter a description\.", 5000\)
            return

        QtWidgets\.QApplication\.setOverrideCursor\(QtCore\.Qt\.WaitCursor\)
        self\.statusbar\.showMessage\("Connecting to Ollama and generating tags\.\.\."\)
        self\.generate_from_desc_btn\.setEnabled\(False\)

        try:
            tagger = get_description_tagger\(\)
            self\.statusbar\.showMessage\("Checking Ollama connection\.\.\."\)
            
            if not tagger\.check_connection\(\):
                raise RuntimeError\(
                    "❌ Cannot connect to Ollama at http://localhost:11434\\n\\n"
                    "Please make sure Ollama is running:\\n"
                    "1\. Download from ollama\.ai\\n"
                    "2\. Run in terminal: ollama serve\\n"
                    "3\. In another terminal: ollama pull qwen2:7b"
                \)

            available_models = tagger\.list_available_models\(\)
            if not available_models:
                raise RuntimeError\(
                    "❌ No models found in Ollama\.\\n\\n"
                    "Download one with:\\n"
                    "  ollama pull qwen2:7b"
                \)

            self\.statusbar\.showMessage\("Generating tags from description\.\.\."\)
            result = tagger\.generate_tags\(description\)

            # Display tags in the dedicated area
            tags_output = "\\n"\.join\(result\.tags\)
            self\.desc_tags_display\.setPlainText\(
                f"Generated {len\(result\.tags\)} tags:\\n\\n{tags_output}"
            \)
            
            self\.statusbar\.showMessage\(f"✓ Generated {len\(result\.tags\)} tags from description\.", 5000\)

        except RuntimeError as e:
            self\.desc_tags_display\.setPlainText\(f"⚠️ Error:\\n\\n{str\(e\)}"\)
            self\.statusbar\.showMessage\("Tag generation failed\.", 5000\)
        except Exception as e:
            self\.desc_tags_display\.setPlainText\(f"⚠️ Unexpected error:\\n\\n{str\(e\)}"\)
            self\.statusbar\.showMessage\("Tag generation error\.", 5000\)
        finally:
            QtWidgets\.QApplication\.restoreOverrideCursor\(\)
            self\.generate_from_desc_btn\.setEnabled\(True\)'''

new_method = '''    def _refresh_available_models(self) -> None:
        """Refresh the list of available Ollama models."""
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        
        try:
            tagger = get_description_tagger()
            if not tagger.check_connection():
                self.model_selector.addItem("❌ Ollama not running", None)
                self.statusbar.showMessage("Ollama is not running. Start with: ollama serve", 5000)
            else:
                models = tagger.list_available_models()
                if models:
                    for model in models:
                        self.model_selector.addItem(model, model)
                    self.statusbar.showMessage(f"Found {len(models)} model(s)", 3000)
                else:
                    self.model_selector.addItem("(No models found)", None)
                    self.statusbar.showMessage("No models found. Run: ollama pull qwen2:7b", 5000)
        except Exception as e:
            self.model_selector.addItem(f"❌ Error: {str(e)[:30]}", None)
            self.statusbar.showMessage(f"Failed to load models: {str(e)}", 5000)
        finally:
            self.model_selector.blockSignals(False)

    def _generate_tags_from_description(self) -> None:
        """Generate Danbooru tags from text description using Ollama."""
        description = self.description_input.toPlainText().strip()
        if not description:
            self.statusbar.showMessage("Description is empty. Please enter a description.", 5000)
            return

        selected_model = self.model_selector.currentData()
        if not selected_model:
            self.statusbar.showMessage("No model selected. Refresh models or start Ollama.", 5000)
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusbar.showMessage(f"Generating tags with {selected_model}...")
        self.generate_from_desc_btn.setEnabled(False)

        try:
            tagger = get_description_tagger(model=selected_model)
            self.statusbar.showMessage("Checking Ollama connection...")
            
            if not tagger.check_connection():
                raise RuntimeError(
                    "❌ Cannot connect to Ollama at http://localhost:11434\n\n"
                    "Please make sure Ollama is running:\n"
                    "1. Download from ollama.ai\n"
                    "2. Run in terminal: ollama serve"
                )

            self.statusbar.showMessage(f"Generating tags using {selected_model}...")
            result = tagger.generate_tags(description)

            # Display tags in the dedicated area
            tags_output = "\n".join(result.tags)
            self.desc_tags_display.setPlainText(
                f"✓ Generated {len(result.tags)} tags with {selected_model}:\n\n{tags_output}"
            )
            
            self.statusbar.showMessage(f"✓ Generated {len(result.tags)} tags from description.", 5000)

        except RuntimeError as e:
            self.desc_tags_display.setPlainText(f"⚠️ Error:\n\n{str(e)}")
            self.statusbar.showMessage("Tag generation failed.", 5000)
        except Exception as e:
            self.desc_tags_display.setPlainText(f"⚠️ Unexpected error:\n\n{str(e)}")
            self.statusbar.showMessage("Tag generation error.", 5000)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.generate_from_desc_btn.setEnabled(True)'''

# Use simple string matching instead of regex since the method is too complex for regex
start_marker = "    def _generate_tags_from_description(self) -> None:"
end_marker = "    def export_caption(self) -> None:"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_method + "\n\n" + content[end_idx:]
    print("✓ Replaced _generate_tags_from_description method")

with open('frontend/native_app.py', 'w') as f:
    f.write(content)

print("✓ All fixes applied!")
