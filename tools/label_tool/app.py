import csv
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

DATASET_PATH = Path(
    os.getenv(
        "DATASET_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "cleaned_tweets_for_labelling.csv"),
    )
)
LABELS_PATH = Path(
    os.getenv(
        "LABELS_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "Depression_labels.csv"),
    )
)
VALID_LABELS = {"Depresi", "Tidak Depresi"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(TOOL_DIR / "templates"))
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    secret_key = os.urandom(24)
    logger.warning("SECRET_KEY is not set; using an ephemeral random key for this session.")
app.secret_key = secret_key

DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded dataset with {len(df)} records")
except FileNotFoundError:
    logger.error(f"Dataset file not found at {DATASET_PATH}")
    df = pd.DataFrame(columns=["full_text"])


def initialize_labeling():
    """Load existing labels and set current index."""
    if LABELS_PATH.exists():
        try:
            existing_labels = pd.read_csv(LABELS_PATH)
            session["current_index"] = len(existing_labels)
            logger.info(f"Resuming from index {session['current_index']}")
            return True
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            session["current_index"] = 0
            return False
    session["current_index"] = 0
    return True


@app.before_request
def before_request():
    """Initialize session variables."""
    if "current_index" not in session:
        initialize_labeling()


def is_labelling_complete():
    return session["current_index"] >= len(df)


def save_label(text, label):
    """Append single label to CSV."""
    try:
        file_exists = LABELS_PATH.is_file() and LABELS_PATH.stat().st_size > 0
        with LABELS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["text", "label"])
            writer.writerow([text, label])
        return True
    except Exception as e:
        logger.error(f"Error saving label: {e}")
        return False


@app.route("/")
def index():
    if is_labelling_complete():
        return render_template("complete.html", total=len(df))

    current_index = session["current_index"]
    total = len(df)
    progress = f"{current_index} out of {total} tweets labeled"

    return render_template(
        "labeling.html",
        text=df["full_text"].iloc[current_index],
        progress=progress,
        current_index=current_index,
        total=total,
    )


@app.route("/label", methods=["POST"])
def label():
    label_val = request.form["label"]
    if label_val not in VALID_LABELS:
        return "Invalid label. Please try again.", 400

    current_index = session["current_index"]
    try:
        text = df["full_text"].iloc[current_index]
        if save_label(text, label_val):
            session["current_index"] = current_index + 1
    except Exception as e:
        logger.error(f"Labeling error: {e}")
        return "Error processing label", 500

    if is_labelling_complete():
        return redirect(url_for("complete"))
    return redirect(url_for("index"))


@app.route("/complete")
def complete():
    return render_template("complete.html", total=len(df))


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", os.getenv("DEBUG", "False")).lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)
