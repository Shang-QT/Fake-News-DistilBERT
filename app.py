# ============================================================
# app.py — Philippine Fake News Detector
# DistilBERT + Tesseract OCR | Streamlit App
# ============================================================

import os
import re
import sys
import unicodedata
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import pytesseract
from PIL import Image
from transformers import DistilBertModel, DistilBertTokenizer

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "PH Fake News Detector",
    page_icon  = "🇵🇭",
    layout     = "centered"
)

# ── Constants ─────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256
DEVICE      = torch.device("cpu")

# ── Google Drive File IDs ─────────────────────────────────────
MODEL_FILE_ID   = "1WSicJ_qp2FQC5d3e1aT-1qePjNQsWd1L"
WEIGHTS_FILE_ID = "1bKrQf_GhoPl6O37EQVrTNpJc7lcY01Qh"

MODEL_PATH   = Path("artifacts/models/distilbert_ph_fakenews.pth")
WEIGHTS_PATH = Path("artifacts/checkpoints/class_weights.pt")

# ── Model definition ──────────────────────────────────────────
class DistilBertClassifier(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs    = self.distilbert(
                         input_ids      = input_ids,
                         attention_mask = attention_mask
                     )
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropped    = self.dropout(cls_output)
        return self.classifier(dropped)

# ── Download from Google Drive ────────────────────────────────
def download_file(file_id, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with st.spinner(f"Downloading {output_path.name} from Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=False)
            st.success(f"✓ {output_path.name} downloaded!")

# ── Load model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    download_file(MODEL_FILE_ID,   MODEL_PATH)
    download_file(WEIGHTS_FILE_ID, WEIGHTS_PATH)

    tokenizer     = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    class_weights = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    if not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(
            class_weights, dtype=torch.float32
        )

    model = DistilBertClassifier(dropout=0.4)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    return model, tokenizer

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x0c", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\b(?:https?://|www\.)\S+\b", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"\\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# ── Predict with chunking ─────────────────────────────────────
def predict_text(text, model, tokenizer, max_chunks=5):
    if not text or str(text).strip() == "":
        return {
            "prediction" : "UNKNOWN",
            "confidence" : 0.0,
            "reason"     : "Empty text",
            "input"      : ""
        }

    cleaned = clean_text(text)
    tokens  = tokenizer(
        cleaned,
        add_special_tokens = False,
        return_tensors     = "pt"
    )
    input_ids_full = tokens["input_ids"][0]
    chunk_size     = MAX_LENGTH - 2
    chunks         = [
        input_ids_full[i:i + chunk_size]
        for i in range(0, len(input_ids_full), chunk_size)
    ]
    chunks    = chunks[:max_chunks]
    all_probs = []

    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            chunk_ids = torch.cat([
                torch.tensor([tokenizer.cls_token_id]),
                chunk,
                torch.tensor([tokenizer.sep_token_id])
            ]).unsqueeze(0).to(DEVICE)

            chunk_mask = torch.ones_like(chunk_ids).to(DEVICE)
            logits     = model(chunk_ids, chunk_mask)
            probs      = torch.softmax(logits, dim=1)
            all_probs.append(probs)

    avg_probs  = torch.mean(torch.cat(all_probs, dim=0), dim=0)
    pred_class = torch.argmax(avg_probs).item()
    confidence = avg_probs[pred_class].item()
    label      = "CREDIBLE" if pred_class == 0 else "NOT CREDIBLE"

    return {
        "prediction"  : label,
        "confidence"  : round(confidence * 100, 2),
        "class_id"    : pred_class,
        "chunks_used" : len(chunks),
        "input"       : cleaned[:80] + "..."
    }

# ── OCR extraction ────────────────────────────────────────────
def extract_text(image):
    return pytesseract.image_to_string(
        image,
        lang   = "eng",
        config = "--oem 3 --psm 6"
    )

# ── Display result ────────────────────────────────────────────
def show_result(result):
    if result["prediction"] == "UNKNOWN":
        st.warning("⚠️ Could not analyze — text too short or empty.")
        return

    confidence = result["confidence"]

    if result["prediction"] == "CREDIBLE":
        st.success("✅ CREDIBLE")
    elif confidence < 70:
        st.warning("⚠️ UNCERTAIN — Low confidence, needs human review")
    else:
        st.error("🚨 NOT CREDIBLE")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence",  f"{confidence:.2f}%")
    col2.metric("Chunks Used", result["chunks_used"])
    col3.metric("Status",      result["prediction"])

    if confidence < 70:
        st.info(
            "💡 Low confidence detected. Try providing "
            "more article text for a more accurate result."
        )

# ────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────
st.title("🇵🇭 Philippine Fake News Detector")
st.caption("DistilBERT + Tesseract OCR · 4th Year CS Thesis")
st.divider()

# Tips expander
with st.expander("💡 Tips for best results"):
    st.markdown("""
    - Paste **at least 3 paragraphs** of article text
    - For screenshots, **crop to article body only**
    - Remove browser toolbars, sidebars, and comment sections
    - Avoid screenshots with social media UI elements
    - Longer text = more accurate prediction
    """)

# Load model
with st.spinner("Loading model..."):
    model, tokenizer = load_model()
st.success("✅ Model ready!")
st.divider()

# Input tabs
tab1, tab2 = st.tabs(["📝 Paste Text", "🖼️ Upload Screenshot"])

# ── Tab 1: Text input ─────────────────────────────────────────
with tab1:
    st.subheader("Paste a news article")
    user_text = st.text_area(
        label            = "Article text",
        placeholder      = "Paste a Philippine news article here...",
        height           = 250,
        label_visibility = "collapsed"
    )

    word_count = len(user_text.split()) if user_text else 0
    st.caption(f"Word count: {word_count}")

    if word_count < 30 and word_count > 0:
        st.warning("⚠️ Text is very short — results may be inaccurate.")

    if st.button("🔍 Analyze Text", use_container_width=True, key="btn_text"):
        if not user_text.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_text(user_text, model, tokenizer)
            show_result(result)

            with st.expander("🔎 Cleaned text sent to model"):
                st.text(clean_text(user_text)[:500])

# ── Tab 2: Image upload ───────────────────────────────────────
with tab2:
    st.subheader("Upload a news screenshot")
    st.caption("Crop to article text only for best results")

    uploaded = st.file_uploader(
        label            = "Upload screenshot",
        type             = ["png", "jpg", "jpeg"],
        label_visibility = "collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded screenshot",
                 use_column_width=True)

        if st.button("🔍 Analyze Image",
                     use_container_width=True, key="btn_img"):
            with st.spinner("Running OCR..."):
                raw_text = extract_text(image)

            ocr_words = len(raw_text.split())
            st.caption(f"OCR extracted: {ocr_words} words")

            with st.expander("📄 OCR extracted text"):
                st.text(raw_text[:500])

            if ocr_words < 30:
                st.warning(
                    "⚠️ Very little text extracted. "
                    "Try a clearer or larger screenshot."
                )
            else:
                with st.spinner("Analyzing..."):
                    result = predict_text(raw_text, model, tokenizer)
                result["ocr_words"] = ocr_words
                show_result(result)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Philippine Fake News Detection System · "
    "DistilBERT + Tesseract OCR · "
    "4th Year CS Thesis · Davao City"
)