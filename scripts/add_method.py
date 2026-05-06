#!/usr/bin/env python3
# Add the _refresh_available_models method

with open('frontend/native_app.py', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Find the line with "def _generate_tags_from_description"
insert_idx = None
for i, line in enumerate(lines):
    if 'def _generate_tags_from_description' in line:
        insert_idx = i
        break

if insert_idx:
    new_method = '''    def _refresh_available_models(self) -> None:
        """Refresh the list of available Ollama models."""
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        try:
            tagger = get_description_tagger()
            if not tagger.check_connection():
                self.model_selector.addItem("(Ollama not running)", None)
            else:
                for model in tagger.list_available_models():
                    self.model_selector.addItem(model, model)
        except:
            self.model_selector.addItem("(error loading)", None)
        finally:
            self.model_selector.blockSignals(False)

'''
    lines.insert(insert_idx, new_method)
    
    with open('frontend/native_app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("✓ Added _refresh_available_models method")
else:
    print("❌ Could not find insertion point")
