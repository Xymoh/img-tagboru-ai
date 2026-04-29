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


st.set_page_config(page_title="Local Anime Tagger", page_icon="🏷️", layout="wide")


def _default_sort_frame(predictions) -> pd.DataFrame:
    rows = []
    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "include": True,
                "rank": index + 1,
                "tag": prediction.tag,
                "confidence": round(prediction.confidence, 4),
                "category": category_label(prediction.category),
            }
        )
    return pd.DataFrame(rows)


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


def _apply_filters(df: pd.DataFrame, blacklist: list[str], whitelist: list[str]) -> pd.DataFrame:
    working = df.copy()
    blacklist_set = {item.strip().lower() for item in blacklist if item.strip()}
    whitelist_set = {item.strip().lower() for item in whitelist if item.strip()}

    def keep_row(tag: str) -> bool:
        normalized = str(tag).strip().lower()
        if blacklist_set and normalized in blacklist_set:
            return False
        if whitelist_set and normalized not in whitelist_set:
            return False
        return True

    working["include"] = working["include"] & working["tag"].map(keep_row)
    return working


def _caption_from_frame(df: pd.DataFrame) -> str:
    included = df[df["include"]].copy()
    if "rank" in included.columns:
        included = included.sort_values(by=["rank", "confidence"], ascending=[True, False])
    return ", ".join(included["tag"].tolist())


def _export_zip(results: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in results:
            archive.writestr(item["caption_name"], item["caption"].encode("utf-8"))
    return buffer.getvalue()


st.title("Local Anime Image Tagger")
st.caption("WD14-style local tagging for anime images, captions, and LoRA training prep.")

with st.sidebar:
    st.header("Tagging Settings")
    general_threshold = st.slider("General tag threshold", 0.0, 1.0, 0.35, 0.01)
    character_threshold = st.slider("Character tag threshold", 0.0, 1.0, 0.85, 0.01)
    normalize_pixels = st.checkbox("Normalize pixels to 0-1", value=False)
    use_mcut = st.checkbox("Use MCut thresholding", value=False)
    max_tags = st.slider("Max tags per image", 5, 200, 80, 5)
    sort_mode = st.selectbox("Sort tags by", ["confidence", "alphabetical", "manual rank"], index=0)
    category_choice = st.multiselect(
        "Show categories",
        ["general", "character"],
        default=["general", "character"],
    )
    blacklist_text = st.text_area("Blacklist", placeholder="tag1, tag2")
    whitelist_text = st.text_area("Whitelist", placeholder="tag1, tag2 (optional)")
    include_scores = st.checkbox("Show scores in caption text", value=False)

st.subheader("Input")
input_mode = st.radio("Choose input source", ["Upload images", "Process local folder"], horizontal=True)

uploaded_files = []
folder_path = ""
if input_mode == "Upload images":
    uploaded_files = st.file_uploader("Drop anime images here", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True)
else:
    folder_path = st.text_input("Folder path", placeholder=r"C:\path\to\images")

process_clicked = st.button("Tag images", type="primary")

if "results" not in st.session_state:
    st.session_state["results"] = []

if process_clicked:
    allowed_category_labels = set(category_choice)
    items: list[dict] = []

    if input_mode == "Upload images" and uploaded_files:
        progress = st.progress(0)
        for index, uploaded in enumerate(uploaded_files, start=1):
            image_bytes = uploaded.getvalue()
            image = image_from_bytes(image_bytes)
            predictions = _tag_image(image, general_threshold, character_threshold, normalize_pixels, use_mcut, max_tags)
            frame = _default_sort_frame(predictions)
            frame = frame[frame["category"].isin(allowed_category_labels)].copy()
            frame = _apply_filters(
                frame,
                [tag.strip() for tag in blacklist_text.split(",")],
                [tag.strip() for tag in whitelist_text.split(",")],
            )
            if sort_mode == "alphabetical":
                frame = frame.sort_values(by=["tag", "confidence"], ascending=[True, False])
            elif sort_mode == "confidence":
                frame = frame.sort_values(by=["confidence", "tag"], ascending=[False, True])
            caption = _caption_from_frame(frame)
            if include_scores:
                caption = ", ".join(
                    f"{row.tag}:{row.confidence:.3f}" for row in frame[frame["include"]].itertuples()
                )
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
    else:
        path = Path(folder_path)
        if path.exists() and path.is_dir():
            image_paths = sorted(
                [
                    file_path
                    for file_path in path.rglob("*")
                    if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                ]
            )
            progress = st.progress(0)
            for index, image_path in enumerate(image_paths, start=1):
                image = Image.open(image_path).convert("RGB")
                predictions = _tag_image(image, general_threshold, character_threshold, normalize_pixels, use_mcut, max_tags)
                frame = _default_sort_frame(predictions)
                frame = frame[frame["category"].isin(allowed_category_labels)].copy()
                frame = _apply_filters(
                    frame,
                    [tag.strip() for tag in blacklist_text.split(",")],
                    [tag.strip() for tag in whitelist_text.split(",")],
                )
                if sort_mode == "alphabetical":
                    frame = frame.sort_values(by=["tag", "confidence"], ascending=[True, False])
                elif sort_mode == "confidence":
                    frame = frame.sort_values(by=["confidence", "tag"], ascending=[False, True])
                caption = _caption_from_frame(frame)
                if include_scores:
                    caption = ", ".join(
                        f"{row.tag}:{row.confidence:.3f}" for row in frame[frame["include"]].itertuples()
                    )
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
        else:
            st.error("Enter a valid folder path or upload at least one image.")

    st.session_state["results"] = items

results = st.session_state["results"]

if results:
    st.subheader("Results")
    tab_labels = [item["name"] for item in results]
    tabs = st.tabs(tab_labels)

    for index, (tab, item) in enumerate(zip(tabs, results)):
        with tab:
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.image(item["image"], caption=item["name"], use_container_width=True)
            with col_right:
                edited_frame = st.data_editor(
                    item["frame"],
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "include": st.column_config.CheckboxColumn("Include"),
                        "rank": st.column_config.NumberColumn("Rank", min_value=1, step=1),
                        "tag": st.column_config.TextColumn("Tag"),
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                        "category": st.column_config.TextColumn("Category"),
                    },
                    key=f"editor_{index}",
                )
                edited_frame = edited_frame.copy()
                edited_frame["include"] = edited_frame["include"].fillna(False)
                edited_frame["rank"] = edited_frame["rank"].fillna(9999)
                if sort_mode == "alphabetical":
                    edited_frame = edited_frame.sort_values(by=["tag", "confidence"], ascending=[True, False])
                elif sort_mode == "confidence":
                    edited_frame = edited_frame.sort_values(by=["confidence", "tag"], ascending=[False, True])
                else:
                    edited_frame = edited_frame.sort_values(by=["rank", "confidence"], ascending=[True, False])

                caption = _caption_from_frame(edited_frame)
                if include_scores:
                    caption = ", ".join(
                        f"{row.tag}:{row.confidence:.3f}" for row in edited_frame[edited_frame["include"]].itertuples()
                    )
                item["frame"] = edited_frame
                item["caption"] = caption
                caption_key = f"caption_{index}"
                st.session_state[caption_key] = caption
                st.text_area("Caption", value=caption, height=140, key=caption_key)

    st.divider()
    captions_text = "\n".join(f"{item['caption_name']}: {item['caption']}" for item in results)
    st.download_button("Download captions as zip", _export_zip(results), file_name="captions.zip", mime="application/zip")
    st.text_area("Batch summary", value=captions_text, height=180)

    output_folder = st.text_input("Optional output folder", placeholder=r"C:\path\to\save\captions")
    if st.button("Save .txt files to folder"):
        destination = Path(output_folder)
        if destination.exists() and destination.is_dir():
            for item in results:
                (destination / item["caption_name"]).write_text(item["caption"], encoding="utf-8")
            st.success(f"Saved {len(results)} caption files.")
        else:
            st.error("Please enter a valid output folder path.")
