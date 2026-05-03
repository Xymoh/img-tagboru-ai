from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tagger import category_label, get_tagger, image_from_bytes, predict_tags
from backend.tag_utils import (
    IMAGE_EXTENSIONS,
    apply_filters,
    caption_from_frame,
    export_zip,
    frame_from_predictions,
    sort_frame,
)


# ---------------------------------------------------------------------------
# Configure Streamlit page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local Anime Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .tip-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
    }
    h2, h3 { color: #1e3a8a; font-weight: 600; }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .css-1d391kg { background-color: #f8fafc; }
    .dataframe { border-radius: 5px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Application-specific tag helpers (thin wrappers)
# ---------------------------------------------------------------------------


def _tag_image(
    image: Image.Image,
    general_threshold: float,
    character_threshold: float,
    normalize_pixels: bool,
    use_mcut: bool,
    limit: int,
):
    tagger = get_tagger()
    return predict_tags(
        tagger,
        image,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        normalize_pixels=normalize_pixels,
        use_mcut=use_mcut,
        limit=limit if limit > 0 else None,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1 style="margin: 0; font-size: 2rem;">🏷️ Local Anime Image Tagger</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
        WD14-style local tagging for anime images, captions, and LoRA training preparation
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Usage Instructions
with st.expander("📖 How to Use This Tool", expanded=False):
    st.markdown(
        """
<div class="info-box">
    <h4 style="margin-top: 0;">Quick Start Guide</h4>
    <ol>
        <li><strong>Configure Settings:</strong> Adjust the tagging thresholds and options in the sidebar</li>
        <li><strong>Load Images:</strong> Either upload images directly or specify a folder path</li>
        <li><strong>Tag Images:</strong> Click "Tag images" to generate tags using the WD14 tagger model</li>
        <li><strong>Review & Edit:</strong> Use the interactive table to include/exclude tags and adjust rankings</li>
        <li><strong>Export:</strong> Download captions as a ZIP file or save them to a folder</li>
    </ol>
</div>

<div class="tip-box">
    <h4 style="margin-top: 0;">💡 Tips for Better Results</h4>
    <ul>
        <li><strong>General Threshold:</strong> Lower values (0.25-0.35) catch more tags but may include irrelevant ones</li>
        <li><strong>Character Threshold:</strong> Keep higher (0.8-0.9) to avoid false character detections</li>
        <li><strong>Blacklist:</strong> Add tags you never want (e.g., "blurry, lowres, bad_anatomy")</li>
        <li><strong>Whitelist:</strong> If you only want specific tags, add them here</li>
        <li><strong>Max Tags:</strong> Limit tags per image to keep captions clean (40-80 is typical)</li>
    </ul>
</div>

<div class="info-box">
    <h4 style="margin-top: 0;">📋 Output Formats</h4>
    <ul>
        <li><strong>Caption Text:</strong> Comma-separated tags ready for training (e.g., "1girl, smile, blue_eyes")</li>
        <li><strong>With Scores:</strong> Include confidence scores (e.g., "1girl:0.98, smile:0.95")</li>
        <li><strong>TXT Files:</strong> Each image gets a matching .txt file with its caption</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Tagging Settings")

    with st.container():
        st.markdown("**Threshold Settings**")
        general_threshold = st.slider(
            "General tag threshold",
            0.0,
            1.0,
            0.6,
            0.01,
            help="Lower = more tags (may include false positives). Recommended: 0.25-0.40",
        )
        character_threshold = st.slider(
            "Character threshold",
            0.0,
            1.0,
            0.85,
            0.01,
            help="Higher = only confident character matches. Recommended: 0.80-0.95",
        )

    with st.container():
        st.markdown("**Processing Options**")
        normalize_pixels = st.checkbox(
            "Normalize pixels to 0-1",
            value=False,
            help="Standardize pixel values. Usually not needed for WD14 models",
        )
        use_mcut = st.checkbox(
            "Use MCut thresholding",
            value=False,
            help="Automatic threshold detection. Overrides manual threshold if enabled",
        )
        max_tags = st.slider(
            "Max tags per image",
            5,
            200,
            40,
            5,
            help="Limit total tags to keep captions manageable",
        )

    with st.container():
        st.markdown("**Display Options**")
        sort_mode = st.selectbox(
            "Sort tags by",
            ["confidence", "alphabetical", "manual rank"],
            index=0,
            help="How to order tags in the results table",
        )
        category_choice = st.multiselect(
            "Show categories",
            ["general", "character"],
            default=["general", "character"],
            help="Filter which tag categories to display",
        )

    with st.container():
        st.markdown("**Filter Tags**")
        blacklist_text = st.text_area(
            "Blacklist",
            placeholder="tag1, tag2",
            help="Tags to exclude (comma-separated). Example: blurry, lowres, bad_anatomy",
        )
        whitelist_text = st.text_area(
            "Whitelist",
            placeholder="tag1, tag2 (optional)",
            help="Only include these tags if specified (comma-separated)",
        )
        include_scores = st.checkbox(
            "Show scores in caption",
            value=False,
            help="Include confidence scores in exported captions",
        )

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
st.markdown("### 📁 Input")
input_mode = st.radio(
    "Choose input source",
    ["Upload images", "Process local folder"],
    horizontal=True,
    help="Upload individual files or process an entire folder",
)

uploaded_files = []
folder_path = ""
if input_mode == "Upload images":
    uploaded_files = st.file_uploader(
        "Drop anime images here",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        help="Select one or more image files to tag",
    )
else:
    folder_path = st.text_input(
        "Folder path",
        placeholder=r"C:\path\to\images",
        help="Full path to folder containing images",
    )

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    process_clicked = st.button("🏷️ Tag Images", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = []

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if process_clicked:
    if not uploaded_files and input_mode == "Upload images":
        st.warning("⚠️ Please upload at least one image.")
        st.stop()
    if not folder_path and input_mode == "Process local folder":
        st.warning("⚠️ Please enter a folder path.")
        st.stop()

    allowed_category_labels = set(category_choice)
    items: list[dict] = []
    blacklist = [tag.strip() for tag in blacklist_text.split(",")]
    whitelist = [tag.strip() for tag in whitelist_text.split(",")]

    if input_mode == "Upload images" and uploaded_files:
        progress = st.progress(0)
        status_text = st.empty()
        for index, uploaded in enumerate(uploaded_files, start=1):
            status_text.text(f"Processing {uploaded.name}... ({index}/{len(uploaded_files)})")
            image_bytes = uploaded.getvalue()
            image = image_from_bytes(image_bytes)
            predictions = _tag_image(image, general_threshold, character_threshold, normalize_pixels, use_mcut, max_tags)
            frame = frame_from_predictions(predictions)
            frame = frame[frame["category"].isin(allowed_category_labels)].copy()
            frame = apply_filters(frame, blacklist, whitelist)
            frame = sort_frame(frame, sort_mode)
            caption = caption_from_frame(frame, include_scores=include_scores)
            items.append(
                {
                    "name": uploaded.name,
                    "image": image,
                    "frame": frame,
                    "caption": caption,
                    "caption_name": f"{Path(uploaded.name).stem}.txt",
                }
            )
            progress.progress(index / len(uploaded_files))
        status_text.empty()
    else:
        path = Path(folder_path)
        if path.exists() and path.is_dir():
            image_paths = sorted(
                [fp for fp in path.rglob("*") if fp.suffix.lower() in IMAGE_EXTENSIONS]
            )
            if not image_paths:
                st.warning("⚠️ No images found in the specified folder.")
                st.stop()
            progress = st.progress(0)
            status_text = st.empty()
            for index, image_path in enumerate(image_paths, start=1):
                status_text.text(f"Processing {image_path.name}... ({index}/{len(image_paths)})")
                image = Image.open(image_path).convert("RGB")
                predictions = _tag_image(image, general_threshold, character_threshold, normalize_pixels, use_mcut, max_tags)
                frame = frame_from_predictions(predictions)
                frame = frame[frame["category"].isin(allowed_category_labels)].copy()
                frame = apply_filters(frame, blacklist, whitelist)
                frame = sort_frame(frame, sort_mode)
                caption = caption_from_frame(frame, include_scores=include_scores)
                items.append(
                    {
                        "name": image_path.name,
                        "image": image,
                        "frame": frame,
                        "caption": caption,
                        "caption_name": f"{image_path.stem}.txt",
                    }
                )
                progress.progress(index / max(1, len(image_paths)))
            status_text.empty()
        else:
            st.error("❌ Enter a valid folder path or upload at least one image.")
            st.stop()

    st.session_state["results"] = items
    st.success(f"✅ Successfully processed {len(items)} image(s)!")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
results = st.session_state["results"]

if results:
    st.markdown("### 📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Processed", len(results))
    with col2:
        total_tags = sum(len(item["frame"][item["frame"]["include"]]) for item in results)
        st.metric("Total Tags", total_tags)
    with col3:
        avg_tags = total_tags / len(results) if results else 0
        st.metric("Avg Tags/Image", f"{avg_tags:.1f}")
    with col4:
        st.metric("Categories", len(category_choice))

    st.markdown("---")

    tab_labels = [item["name"] for item in results]
    tabs = st.tabs([f"🖼️ {name}" for name in tab_labels])

    for index, (tab, item) in enumerate(zip(tabs, results)):
        with tab:
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.image(item["image"], caption=item["name"], use_container_width=True)
                st.caption(f"File: {item['name']}")
            with col_right:
                st.markdown("**Edit Tags**")
                st.caption("Uncheck tags to exclude them from the caption")
                edited_frame = st.data_editor(
                    item["frame"],
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "include": st.column_config.CheckboxColumn("Include", help="Uncheck to exclude this tag"),
                        "rank": st.column_config.NumberColumn("Rank", min_value=1, step=1, help="Manual ranking order"),
                        "tag": st.column_config.TextColumn("Tag", help="Tag name"),
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.4f", help="Model confidence"),
                        "category": st.column_config.TextColumn("Category", help="Tag category (general/character)"),
                    },
                    key=f"editor_{index}",
                )
                edited_frame = edited_frame.copy()
                edited_frame["include"] = edited_frame["include"].fillna(False)
                edited_frame["rank"] = edited_frame["rank"].fillna(9999)
                edited_frame = sort_frame(edited_frame, sort_mode)

                caption = caption_from_frame(edited_frame, include_scores=include_scores)
                item["frame"] = edited_frame
                item["caption"] = caption
                caption_key = f"caption_{index}"
                st.session_state[caption_key] = caption

                st.markdown("**Generated Caption**")
                st.text_area(
                    "Caption",
                    value=caption,
                    height=140,
                    key=caption_key,
                    help="This caption will be saved to the .txt file",
                )
                st.caption(f"Tag count: {len(edited_frame[edited_frame['include']])}")

    st.markdown("---")

    # Export section
    st.markdown("### 💾 Export")
    col_export1, col_export2 = st.columns(2)

    with col_export1:
        captions_text = "\n".join(f"{item['caption_name']}: {item['caption']}" for item in results)
        zip_bytes = export_zip((item["caption_name"], item["caption"]) for item in results)
        st.download_button(
            "📦 Download All Captions (ZIP)",
            zip_bytes,
            file_name="captions.zip",
            mime="application/zip",
            use_container_width=True,
        )
        with st.expander("📄 Batch Summary", expanded=False):
            st.text_area("All captions", value=captions_text, height=180, label_visibility="collapsed")

    with col_export2:
        st.markdown("**Save to Folder**")
        output_folder = st.text_input(
            "Output folder path",
            placeholder=r"C:\path\to\save\captions",
            help="Captions will be saved as .txt files with the same name as images",
        )
        if st.button("💾 Save .txt Files", use_container_width=True):
            if output_folder:
                destination = Path(output_folder)
                if destination.exists() and destination.is_dir():
                    for item in results:
                        (destination / item["caption_name"]).write_text(item["caption"], encoding="utf-8")
                    st.success(f"✅ Saved {len(results)} caption file(s) to {destination}")
                else:
                    st.error("❌ Please enter a valid output folder path.")
            else:
                st.warning("⚠️ Please enter an output folder path.")
