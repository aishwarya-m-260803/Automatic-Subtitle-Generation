from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import whisper
import logging
import librosa
import numpy as np
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'uploads'
SUBTITLE_FOLDER = os.path.join('static', 'subtitles')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB file limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'mp4'}

# Load Whisper model
try:
    model = whisper.load_model("base")  # You can change to "small" or "medium" if needed
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if not model:
        flash("Transcription service unavailable")
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('home'))
    
    if not allowed_file(file.filename):
        flash('Allowed file types: mp3, wav, ogg, m4a, mp4')
        return redirect(url_for('home'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Transcribe with Whisper
        result = model.transcribe(filepath, task="transcribe", verbose=False)
        transcript = result["text"]

        # Generate SRT subtitle file
        srt_filename = filename.rsplit('.', 1)[0] + '.srt'
        srt_path = os.path.join(SUBTITLE_FOLDER, srt_filename)

        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        with open(srt_path, 'w', encoding='utf-8') as f:
            for segment in result['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                f.write(f"{segment['id'] + 1}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text}\n\n")

        # Delete audio file after processing
        os.remove(filepath)

        return render_template('result.html',
                               transcript=transcript,
                               filename=filename,
                               subtitle_file=srt_filename)

    except Exception as e:
        logging.error(f"Error processing {file.filename}: {str(e)}")
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/waveform-data/<filename>')
def get_waveform_data(filename):
    try:
        # Get the subtitle file path
        srt_filename = filename.rsplit('.', 1)[0] + '.srt'
        srt_path = os.path.join(SUBTITLE_FOLDER, srt_filename)
        
        if not os.path.exists(srt_path):
            return jsonify({'error': 'Subtitle file not found'}), 404
        
        # Parse SRT file to get timing data
        segments = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    timing = lines[1]
                    text = ' '.join(lines[2:])
                    
                    # Parse timing (00:00:00,000 --> 00:00:05,440)
                    start_end = timing.split(' --> ')
                    if len(start_end) == 2:
                        start_time = start_end[0]
                        end_time = start_end[1]
                        
                        # Convert to seconds
                        start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.replace(',', '.').split(':'))))
                        end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.replace(',', '.').split(':'))))
                        
                        segments.append({
                            'start': start_seconds,
                            'end': end_seconds,
                            'text': text
                        })
        
        return jsonify({
            'segments': segments,
            'filename': filename
        })
        
    except Exception as e:
        logging.error(f"Error getting waveform data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
