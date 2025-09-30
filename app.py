from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import json
import re
import base64
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import torch
from transformers import pipeline
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models (cache in production)
summarizer = None
translator = None
language_data = []

def init_models():
    global summarizer, translator, language_data
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        print(f"Summarizer loading error: {e}")
        summarizer = None
    
    try:
        translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    except Exception as e:
        print(f"Translator loading error: {e}")
        translator = None
    
    try:
        with open("language.json", "r", encoding="utf-8") as f:
            language_data = json.load(f)
    except Exception as e:
        print(f"Language data loading error: {e}")
        language_data = []

# Initialize on startup
init_models()

def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry.get('Language', '').strip().lower() == language.strip().lower():
            return entry.get('FLORES-200 code')
    return None

def summarize_text(input_text):
    if not summarizer:
        return "Summarizer model not loaded."
    
    if len(input_text) < 2000:
        try:
            out = summarizer(input_text, max_length=150, min_length=30, truncation=True)
            return out[0]["summary_text"]
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', input_text) if p.strip()]
        summaries = []
        for p in paragraphs[:5]:  # Limit to first 5 paragraphs
            pshort = p[:4000]
            try:
                s = summarizer(pshort, max_length=120, min_length=20, truncation=True)[0]["summary_text"]
                summaries.append(s)
            except:
                summaries.append(pshort[:300] + "...")
        return "\n\n".join(summaries)

def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_youtube_transcript_summary(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Could not extract YouTube video ID from the URL."
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        return summarize_text(text_transcript)
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def translate_text_func(text, dest_language):
    if not translator:
        return "Translator model not loaded."
    
    dest_code = get_FLORES_code_from_language(dest_language)
    if not dest_code:
        return f"Destination language '{dest_language}' not found."
    
    try:
        out = translator(text, src_lang="eng_Latn", tgt_lang=dest_code)
        if isinstance(out, list) and len(out) > 0 and "translation_text" in out[0]:
            return out[0]["translation_text"]
        return str(out)
    except Exception as e:
        return f"Translation error: {str(e)}"

def analyze_column(df, column_name):
    col_data = df[column_name].dropna()
    
    stats = {
        "mean": float(col_data.mean()),
        "median": float(col_data.median()),
        "std": float(col_data.std()),
        "min": float(col_data.min()),
        "max": float(col_data.max()),
        "count": int(col_data.count())
    }
    
    # Create plots and convert to base64
    plots = {}
    
    # Histogram
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor='#1a1a2e')
    ax1.set_facecolor('#16213e')
    ax1.hist(col_data, bins=10, color='#0f4c75', edgecolor='#3282b8')
    ax1.set_xlabel(column_name, color='#bbe1fa')
    ax1.set_ylabel("Frequency", color='#bbe1fa')
    ax1.set_title(f"Histogram of {column_name}", color='#bbe1fa')
    ax1.tick_params(colors='#bbe1fa')
    plots['histogram'] = fig_to_base64(fig1)
    plt.close(fig1)
    
    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='#1a1a2e')
    ax2.set_facecolor('#16213e')
    bp = ax2.boxplot(col_data, vert=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#0f4c75')
    ax2.set_xlabel(column_name, color='#bbe1fa')
    ax2.set_title(f"Boxplot of {column_name}", color='#bbe1fa')
    ax2.tick_params(colors='#bbe1fa')
    plots['boxplot'] = fig_to_base64(fig2)
    plt.close(fig2)
    
    # Pie chart
    fig3, ax3 = plt.subplots(figsize=(8, 6), facecolor='#1a1a2e')
    bins = pd.cut(col_data, bins=5)
    pie_data = bins.value_counts().sort_index()
    colors = ['#0f4c75', '#3282b8', '#1b262c', '#0f3460', '#533483']
    ax3.pie(pie_data, labels=[str(i) for i in pie_data.index.astype(str)], 
            autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': '#bbe1fa'})
    ax3.set_title(f"Distribution of {column_name}", color='#bbe1fa')
    plots['piechart'] = fig_to_base64(fig3)
    plt.close(fig3)
    
    return stats, plots

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', facecolor='#1a1a2e')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/news-summarizer')
def news_summarizer_page():
    return render_template('news_summarizer.html')

@app.route('/news-analyzer')
def news_analyzer_page():
    return render_template('news_analyzer.html')

@app.route('/media-summarizer')
def media_summarizer_page():
    return render_template('media_summarizer.html')

@app.route('/translator')
def translator_page():
    languages = [entry.get('Language') for entry in language_data] if language_data else []
    return render_template('translator.html', languages=languages)

# API Endpoints
@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    summary = summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/api/summarize-file', methods=['POST'])
def api_summarize_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        content = file.read().decode('utf-8')
        summary = summarize_text(content)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/youtube-summarize', methods=['POST'])
def api_youtube_summarize():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    summary = get_youtube_transcript_summary(url)
    return jsonify({'summary': summary})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    column = request.form.get('column')
    
    try:
        df = pd.read_excel(file)
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not column:
            return jsonify({'columns': numeric_columns})
        
        if column not in numeric_columns:
            return jsonify({'error': 'Invalid column selected'}), 400
        
        stats, plots = analyze_column(df, column)
        return jsonify({'stats': stats, 'plots': plots})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.get_json()
    text = data.get('text', '')
    dest_language = data.get('language', '')
    
    if not text or not dest_language:
        return jsonify({'error': 'Missing text or language'}), 400
    
    translation = translate_text_func(text, dest_language)
    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
