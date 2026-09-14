from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import csv
from pathlib import Path

load_dotenv()

# Configuration with fallbacks
DATASET_PATH = os.getenv("DATASET_PATH", "C:/Users/mjova/OneDrive/Documents/kuliah/Skripsi/deteksi_depresi/preprocessing/cleaned_tweets_for_labelling.csv")
LABELS_PATH = os.getenv("LABELS_PATH", "C:/Users/mjova/OneDrive/Documents/kuliah/Skripsi/deteksi_depresi/dataset/tweets-data/Depression_labels.csv")
VALID_LABELS = {"Depresi", "Tidak Depresi"}  # Match button values

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")  # Required for sessions

# Create directories if they don't exist
Path(DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(LABELS_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded dataset with {len(df)} records")
except FileNotFoundError:
    logger.error(f"Dataset file not found at {DATASET_PATH}")
    df = pd.DataFrame(columns=['full_text'])  # Create empty DF to prevent crashes

# Initialize labeling state
def initialize_labeling():
    """Load existing labels and set current index"""
    if os.path.exists(LABELS_PATH):
        try:
            existing_labels = pd.read_csv(LABELS_PATH)
            session['current_index'] = len(existing_labels)
            logger.info(f"Resuming from index {session['current_index']}")
            return True
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            session['current_index'] = 0
            return False
    session['current_index'] = 0
    return True

@app.before_request
def before_request():
    """Initialize session variables"""
    if 'current_index' not in session:
        initialize_labeling()

def is_labelling_complete():
    return session['current_index'] >= len(df)

def save_label(text, label):
    """Append single label to CSV"""
    try:
        file_exists = os.path.isfile(LABELS_PATH) and os.path.getsize(LABELS_PATH) > 0
        with open(LABELS_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['text', 'label'])
            writer.writerow([text, label])
        return True
    except Exception as e:
        logger.error(f"Error saving label: {e}")
        return False

@app.route('/')
def index():
    if is_labelling_complete():
        return render_template('complete.html', total=len(df))
    
    current_index = session['current_index']
    total = len(df)
    progress = f"{current_index} out of {total} tweets labeled"

    return render_template('labeling.html', 
                          text=df['full_text'].iloc[current_index],
                          progress=progress,
                          current_index=current_index,
                          total=total)

@app.route('/label', methods=['POST'])
def label():
    label_val = request.form['label']
    if label_val not in VALID_LABELS:
        return "Invalid label. Please try again.", 400
    
    current_index = session['current_index']
    try:
        text = df['full_text'].iloc[current_index]
        if save_label(text, label_val):
            session['current_index'] = current_index + 1
    except Exception as e:
        logger.error(f"Labeling error: {e}")
        return "Error processing label", 500

    if is_labelling_complete():
        return redirect(url_for('complete'))
    return redirect(url_for('index'))

@app.route('/complete')
def complete():
    return render_template('complete.html', total=len(df))

if __name__ == '__main__':
    app.run(debug=os.getenv("DEBUG", "False") == "True")