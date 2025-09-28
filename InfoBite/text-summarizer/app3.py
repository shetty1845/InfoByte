# infobyte_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# transformers / torch
import torch
from transformers import pipeline

# Optional: gradio (we'll keep it available, but Streamlit is the main UI)
import gradio as gr

# ---------- Helper: initialize pipelines (with safe fallbacks) ----------
@st.cache_resource
def init_summarizer():
    try:
        # prefer a hub model; replace with local snapshot path if needed
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.warning(f"Couldn't load summarization model from hub: {e}")
        summarizer = None
    return summarizer

@st.cache_resource
def init_youtube_summarizer():
    # we will use same summarizer; kept for semantic clarity
    return init_summarizer()

@st.cache_resource
def init_translator():
    try:
        # If you have a local model snapshot, point model_path to it.
        # model_path = "../Models/models--facebook--nllb-200-distilled-600M/snapshots/..."
        translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    except Exception as e:
        st.warning(f"Couldn't load translator model from hub: {e}")
        translator = None
    return translator

# ---------- Load FLORES language mapping ----------
@st.cache_data
def load_language_map(json_path="language.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            language_data = json.load(f)
    except Exception as e:
        st.error(f"language.json not found or invalid at {json_path}: {e}")
        return []
    return language_data

def get_FLORES_code_from_language(language, language_data):
    for entry in language_data:
        if entry.get('Language', '').strip().lower() == language.strip().lower():
            return entry.get('FLORES-200 code')
    return None

# ---------- Summarization functions ----------
def summarize_text(input_text, summarizer):
    if not summarizer:
        return "Summarizer model not loaded."
    # chunking guard: summarizers often have max token limits; naive chunking
    if len(input_text) < 2000:
        out = summarizer(input_text, max_length=150, min_length=30, truncation=True)
        return out[0]["summary_text"]
    else:
        # naive chunk summarization: split by paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', input_text) if p.strip()]
        summaries = []
        for p in paragraphs:
            pshort = p[:4000]  # guard
            try:
                s = summarizer(pshort, max_length=120, min_length=20, truncation=True)[0]["summary_text"]
            except Exception:
                s = pshort[:300] + "..."
            summaries.append(s)
        return "\n\n".join(summaries)

def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript_summary(video_url, summarizer):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Could not extract YouTube video id from the URL."
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        return summarize_text(text_transcript, summarizer)
    except Exception as e:
        return f"Error fetching transcript or summarizing: {e}"

# ---------- Translation ----------
def translate_text(text, dest_language, translator, language_data):
    if not translator:
        return "Translator model not loaded."
    dest_code = get_FLORES_code_from_language(dest_language, language_data)
    if not dest_code:
        return f"Destination language '{dest_language}' not found in language.json."
    try:
        out = translator(text, src_lang="eng_Latn", tgt_lang=dest_code)
        # pipeline returns a list with 'translation_text'
        if isinstance(out, list) and len(out) > 0 and "translation_text" in out[0]:
            return out[0]["translation_text"]
        # fallback key
        if isinstance(out, dict) and "translation_text" in out:
            return out["translation_text"]
        return str(out)
    except Exception as e:
        return f"Translation error: {e}"

# ---------- News Analyzer (numerical analysis) ----------
def news_analyzer_from_df(df, selected_col):
    # produce summary stats and figures as matplotlib figures
    col_data = df[selected_col].dropna()
    stats = {
        "mean": float(col_data.mean()),
        "median": float(col_data.median()),
        "std": float(col_data.std()),
        "min": float(col_data.min()),
        "max": float(col_data.max()),
        "count": int(col_data.count())
    }

    # histogram figure
    fig1, ax1 = plt.subplots()
    ax1.hist(col_data, bins=10)
    ax1.set_xlabel(selected_col)
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Histogram of {selected_col}")

    # boxplot figure
    fig2, ax2 = plt.subplots()
    ax2.boxplot(col_data, vert=False)
    ax2.set_xlabel(selected_col)
    ax2.set_title(f"Boxplot of {selected_col}")

    # pie chart (binned)
    fig3, ax3 = plt.subplots()
    bins = pd.cut(col_data, bins=5)
    pie_data = bins.value_counts().sort_index()
    ax3.pie(pie_data, labels=[str(i) for i in pie_data.index.astype(str)], autopct='%1.1f%%', startangle=90)
    ax3.axis('equal')

    return stats, fig1, fig2, fig3

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="InfoByte", layout="wide")
    st.title("InfoByte — News & Media toolkit")
    st.markdown("Use the header to switch between features: **News Summarizer**, **News Analyzer**, **Media Summarizer**, **News Multi-Language Translator**")

    # Header-like selection (tabs or radio)
    mode = st.radio("Choose feature", ["News Summarizer", "News Analyzer", "Media Summarizer", "News Multi-Language Translator"], horizontal=True)

    # Initialize models & data
    summarizer = init_summarizer()
    youtube_summarizer = init_youtube_summarizer()
    translator = init_translator()
    language_data = load_language_map()

    if mode == "News Summarizer":
        st.header("📰 News Summarizer")
        st.info("Paste news text or upload a .txt/.pdf (PDF summarization is plain-text attempt only).")
        col1, col2 = st.columns([3,1])
        with col1:
            text_input = st.text_area("Paste news article or text here:", height=300)
            uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
            if uploaded and not text_input:
                try:
                    t = uploaded.read().decode("utf-8")
                    text_input = t
                except Exception as e:
                    st.error(f"Couldn't read uploaded file: {e}")
        with col2:
            max_len = st.slider("Max summary words (approx)", 50, 500, 150)
            st.write("Model status:")
            st.write("Summarizer loaded" if summarizer else "Summarizer NOT loaded")

        if st.button("Generate Summary"):
            if not text_input:
                st.warning("Please paste or upload text to summarize.")
            else:
                with st.spinner("Summarizing..."):
                    s = summarize_text(text_input, summarizer)
                st.subheader("Summary")
                st.write(s)

    elif mode == "News Analyzer":
        st.header("📊 News Analyzer (numerical analysis)")
        st.info("Upload an Excel (.xlsx) file containing numeric columns (e.g., article lengths, view counts, engagement metrics).")
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if not numeric_columns:
                    st.warning("⚠️ No numeric columns found in the uploaded file.")
                else:
                    selected_column = st.selectbox("Select the numeric column to analyze", numeric_columns)
                    if st.button("Analyze Column"):
                        with st.spinner("Analyzing numerical data..."):
                            stats, fig1, fig2, fig3 = news_analyzer_from_df(df, selected_column)
                        st.subheader("Summary Statistics")
                        st.write(stats)
                        st.subheader("Histogram")
                        st.pyplot(fig1)
                        st.subheader("Boxplot")
                        st.pyplot(fig2)
                        st.subheader("Pie Chart (binned)")
                        st.pyplot(fig3)
            except Exception as e:
                st.error(f"⚠️ Error reading or processing file: {e}")

    elif mode == "Media Summarizer":
        st.header("🎬 Media Summarizer")
        st.info("Summarize YouTube videos (via transcript) or summarize text from uploaded files.")
        yt_url = st.text_input("YouTube URL to summarize (transcript-based):")
        uploaded_file = st.file_uploader("Upload a text file (.txt) to summarize", type=["txt"])
        if st.button("Summarize YouTube"):
            if not yt_url:
                st.warning("Enter a YouTube URL.")
            else:
                with st.spinner("Fetching transcript and summarizing..."):
                    out = get_youtube_transcript_summary(yt_url, youtube_summarizer)
                st.subheader("YouTube Summary")
                st.write(out)

        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                if st.button("Summarize Uploaded File"):
                    with st.spinner("Summarizing file..."):
                        s = summarize_text(content, summarizer)
                    st.subheader("File Summary")
                    st.write(s)
            except Exception as e:
                st.error(f"Couldn't read uploaded file: {e}")

    elif mode == "News Multi-Language Translator":
        st.header("🌐 News Multi-Language Translator")
        st.info("Translate English text to many languages using FLORES codes (language.json).")
        text_to_translate = st.text_area("Enter English text to translate:", height=200)
        langs = [entry.get('Language') for entry in language_data] if language_data else []
        dest_language = st.selectbox("Select Destination Language", langs if langs else ["Hindi", "French", "German"])
        if st.button("Translate"):
            if not text_to_translate:
                st.warning("Please enter text to translate.")
            else:
                with st.spinner("Translating..."):
                    translated = translate_text(text_to_translate, dest_language, translator, language_data)
                st.subheader("Translation")
                st.write(translated)

    # Footer / optionally show small Gradio launch buttons
    st.markdown("---")
    st.caption("InfoByte — built with Streamlit & optional Gradio demos. If you want Gradio interfaces launched for quick testing, uncomment the section at the bottom of this file and run it outside Streamlit (they open separate local web UIs).")

if __name__ == "__main__":
    main()


# ------------------------------
# Optional: Small Gradio demos (uncomment to run separately)
# Note: Gradio creates its own servers; don't run these while running Streamlit in same process.
# ------------------------------
"""
# Simple Gradio translator demo
def gr_translate(text, dest_language):
    language_data = load_language_map()
    translator = init_translator()
    return translate_text(text, dest_language, translator, language_data)

gr_translate_demo = gr.Interface(
    fn=gr_translate,
    inputs=[gr.Textbox(lines=4, label="English text"), gr.Dropdown([e['Language'] for e in load_language_map()], label="Destination language")],
    outputs=[gr.Textbox(lines=4, label="Translated text")],
    title="InfoByte: Quick Translator (Gradio)"
)

# Simple Gradio summarizer demo
def gr_summarize(text):
    summarizer = init_summarizer()
    return summarize_text(text, summarizer)

gr_summarize_demo = gr.Interface(
    fn=gr_summarize,
    inputs=gr.Textbox(lines=6, label="Text to summarize"),
    outputs=gr.Textbox(lines=4, label="Summary"),
    title="InfoByte: Quick Summarizer (Gradio)"
)

# To launch (run each in separate processes):
# gr_translate_demo.launch()
# gr_summarize_demo.launch()
"""
