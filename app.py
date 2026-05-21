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
MODEL_FILE_ID   = "1HmF0iwNRT3Rf-KVgCo0YeIyzqUqGShGe"
WEIGHTS_FILE_ID = "1RW3HjGuCm8aav5_wU2_D7uC7Fhm927Jn"

BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "artifacts" / "models" / "distilbert_ph_fakenews_v3.pth"
WEIGHTS_PATH = BASE_DIR / "artifacts" / "checkpoints" / "class_weights_v3.pt"

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
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(output_path), quiet=False)
                if not output_path.exists():
                    st.error(f"Failed to download {output_path.name}.")
            except Exception as e:
                st.error(f"Exception downloading {output_path.name}: {e}")

# ── Load model (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner=False, hash_funcs={"_main": id})
def load_ml_model():
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
    try:
        return pytesseract.image_to_string(
            image,
            lang   = "eng",
            config = "--oem 3 --psm 6"
        )
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract-OCR is not installed or not in PATH. Please make sure Tesseract is installed for image extraction to work.")
        return ""
    except Exception as e:
        st.error(f"Error during OCR extraction: {e}")
        return ""

# ── Display result ────────────────────────────────────────────
def show_result(result):
    if result["prediction"] == "UNKNOWN":
        st.warning("⚠️ Could not analyze — text too short or empty.")
        return

    confidence = result["confidence"]

    # Styled Result Header
    if result["prediction"] == "CREDIBLE":
        st.success("✅ **Classification: CREDIBLE**")
        color = "green"
    elif confidence < 70:
        st.warning("⚠️ **Classification: UNCERTAIN** — Low confidence, needs human review")
        color = "orange"
    else:
        st.error("🚨 **Classification: NOT CREDIBLE**")
        color = "red"

    # Progress bar for confidence
    st.markdown(f"**Confidence Score:** {confidence:.2f}%")
    st.progress(int(confidence) / 100)

    # Metrics layout
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence",  f"{confidence:.2f}%")
    col2.metric("Chunks Used", result["chunks_used"])
    col3.metric("Prediction",  result["prediction"])

    if confidence < 70:
        st.info(
            "💡 **Low confidence detected.** The model didn't have enough strong signals. "
            "Try providing more of the article text for a more accurate result."
        )

# ────────────────────────────────────────────────────────────
# UI Configuration & Sidebar
# ────────────────────────────────────────────────────────────

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/99/Flag_of_the_Philippines.svg", width=100)
    st.title("About")
    st.info(
        "This tool uses an NLP model (DistilBERT) designed to classify the credibility of Philippine news articles. "
        "It supports both raw text and screenshots via OCR."
    )
    
    st.header("💡 Tips for Best Results")
    st.markdown("""
    - Paste **at least 3 paragraphs** of text
    - Focus on the main body of the article
    - Remove social media captions or sidebars
    - **For images:** Crop purely to the text body
    - Keep sentences continuous
    """)
    st.divider()
    st.caption("Developed with ❤️ using Streamlit & PyTorch")

# Main Header
st.title("Philippine Fake News Detector")
st.markdown("**AI-Powered verification using DistilBERT & Tesseract OCR**")
st.divider()

# Load model
with st.spinner("Loading model resources into memory..."):
    model, tokenizer = load_ml_model()

# Input tabs
tab1, tab2 = st.tabs(["📝 Paste Text Base", "🖼️ Upload Screenshot (OCR)"])

# ── Tab 1: Text input ─────────────────────────────────────────
with tab1:
    st.subheader("Paste a news article")
    
    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""

    def clear_text():
        st.session_state["text_input"] = ""

    user_text = st.text_area(
        label            = "Article text",
        placeholder      = "Paste a Philippine news article here...",
        height           = 250,
        label_visibility = "collapsed",
        key              = "text_input"
    )

    word_count = len(user_text.split()) if user_text else 0
    st.caption(f"Word count: {word_count}")

    if word_count < 30 and word_count > 0:
        st.warning("⚠️ Text is very short — results may be inaccurate.")

    col1, col2 = st.columns(2)
    with col1:
        analyze_text_clicked = st.button("🔍 Analyze Text", use_container_width=True, key="btn_text")
    with col2:
        st.button("🗑️ Clear Text", use_container_width=True, on_click=clear_text)

    if analyze_text_clicked:
        if not user_text.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_text(user_text, model, tokenizer)
            
            st.divider()
            st.subheader("📊 Analysis Results")
            show_result(result)

            with st.expander("🔎 Cleaned text sent to model"):
                st.text(clean_text(user_text)[:500])

# ── Tab 2: Image upload ───────────────────────────────────────
with tab2:
    st.subheader("Upload a news screenshot")
    st.caption("Crop to article text only for best results")

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    def clear_image():
        st.session_state["file_uploader_key"] += 1

    uploaded = st.file_uploader(
        label            = "Upload screenshot",
        type             = ["png", "jpg", "jpeg"],
        label_visibility = "collapsed",
        key              = f"uploader_{st.session_state['file_uploader_key']}"
    )

    if uploaded:
        image = Image.open(uploaded)
        # Using a fixed width so we don't have to scroll past massive images
        st.image(image, caption="Uploaded screenshot", width=400)

        col1, col2 = st.columns(2)
        with col1:
            analyze_img_clicked = st.button("🔍 Analyze Image", use_container_width=True, key="btn_img")
        with col2:
            st.button("🗑️ Clear Image", use_container_width=True, on_click=clear_image)

        if analyze_img_clicked:
            with st.spinner("Running OCR..."):
                raw_text = extract_text(image)

            ocr_words = len(raw_text.split())
            st.caption(f"OCR extracted: {ocr_words} words")

            if ocr_words < 30:
                st.warning(
                    "⚠️ Very little text extracted. "
                    "Try a clearer or larger screenshot."
                )
            else:
                with st.spinner("Analyzing..."):
                    result = predict_text(raw_text, model, tokenizer)
                result["ocr_words"] = ocr_words
                
                st.divider()
                st.subheader("📊 Analysis Results")
                show_result(result)

                with st.expander("📄 OCR extracted text"):
                    st.text(raw_text[:500])

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Philippine Fake News Detection System · "
    "DistilBERT + Tesseract OCR "
)