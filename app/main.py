import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import tensorflow as tf

# Ensure project root is on sys.path whether run via `python app/main.py` or `python -m app.main`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_indo

load_dotenv(PROJECT_ROOT / ".env")

# --- KONFIGURASI PATH & KONSTANTA ---
APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "depression_model_multiclass.h5"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

MAX_LEN = 100
MAX_INPUT_LENGTH = 1000
MIN_POS_HITS = 2
THRESHOLD_SEVERE = 0.35
THRESHOLD_MODERATE = 0.45
CLASSES = {0: "Normal", 1: "Moderate", 2: "Severe"}

POSITIVE_TOKENS = {
    "bahagia", "senang", "semangat", "syukur", "alhamdulillah", "happy", "asik", "seru",
    "positif", "jernih", "nikmat", "menikmati", "tenang", "lega", "baik", "fokus", "stabil",
}
NEGATION_TOKENS = {"tidak", "ga", "gak", "bukan", "tak", "jangan", "nggak", "tdk"}
NEGATIVE_INDICATOR_TOKENS = {
    "sedih", "hampa", "kesepian", "nangis", "putus", "putusasa", "depresi",
    "bunuh", "mati", "akhiri", "pengen", "mau", "beban", "negatif", "hopeless",
}

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

# --- LOAD RESOURCES ---
print("Memuat Model & Tokenizer...")
model = None
tokenizer = None

if MODEL_PATH.exists() and TOKENIZER_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with TOKENIZER_PATH.open("rb") as f:
            tokenizer = pickle.load(f)
        print("Model & Tokenizer berhasil dimuat.")
    except Exception as e:
        print(f"Error loading resources: {e}")
else:
    print(f"PERINGATAN: File model/tokenizer tidak ditemukan di {MODEL_DIR}")


# --- FUNGSI SARAN ---
def get_suggestion(label: str) -> dict[str, str]:
    if label == "Normal":
        return {
            "text": "Kondisi mental Anda terindikasi stabil. Tetap jaga pola hidup sehat dan luangkan waktu untuk hobi.",
            "color": "success",
            "icon": "bi-emoji-smile",
        }
    if label == "Moderate":
        return {
            "text": "Terdeteksi indikasi tekanan emosional. Disarankan untuk istirahat sejenak atau bercerita kepada teman.",
            "color": "warning",
            "icon": "bi-emoji-neutral",
        }
    if label == "Severe":
        return {
            "text": "Terdeteksi indikasi beban emosional berat. Sangat disarankan berkonsultasi dengan profesional.",
            "color": "danger",
            "icon": "bi-exclamation-triangle-fill",
        }
    return {}


# --- LOGIKA INFERENSI & POSITIVE GUARD ---
def evaluate_positive_guard(clean_text: str) -> dict:
    """Evaluate whether preprocessed text expresses a clear positive context without nearby negation."""
    words = clean_text.split()
    pos_count = sum(1 for w in words if w in POSITIVE_TOKENS)
    neg_count = sum(1 for w in words if w in NEGATION_TOKENS)
    negind_count = sum(1 for w in words if w in NEGATIVE_INDICATOR_TOKENS)

    negation_near_pos = False
    for i, w in enumerate(words):
        if w in POSITIVE_TOKENS:
            window = words[max(0, i - 2) : i]
            if any(tok in NEGATION_TOKENS for tok in window):
                negation_near_pos = True
                break

    pos_score = pos_count - (neg_count + negind_count)
    is_positive_context = (
        (pos_count >= MIN_POS_HITS) and (pos_score >= 2) and (not negation_near_pos)
    )
    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "negind_count": negind_count,
        "negation_near_pos": negation_near_pos,
        "is_positive_context": is_positive_context,
    }


def decide_class(probs, is_positive_context: bool) -> tuple[int, bool]:
    """Select final class index using severity thresholds and Positive Guard."""
    if probs[2] > THRESHOLD_SEVERE:
        return 2, False
    if is_positive_context:
        return 0, True
    if probs[1] > THRESHOLD_MODERATE and probs[1] > probs[0]:
        return 1, False
    return 0, False


def predict_text(raw_text: str) -> tuple[dict | None, dict | None, str | None, int]:
    """Validate, preprocess, tokenize, and run depression risk prediction on input text.

    Returns (prediction_result, suggestion, error_message, http_status).
    """
    stripped_text = raw_text.strip()
    if not stripped_text:
        return None, None, "Mohon masukkan teks terlebih dahulu.", 400

    if len(stripped_text) > MAX_INPUT_LENGTH:
        return None, None, f"Teks terlalu panjang (maksimal {MAX_INPUT_LENGTH} karakter).", 400

    if model is None or tokenizer is None:
        return None, None, "Model atau tokenizer belum tersedia di server.", 503

    clean_text = preprocess_indo(raw_text)
    seq = tokenizer.texts_to_sequences([clean_text])
    oov_idx = tokenizer.word_index.get(tokenizer.oov_token)
    if oov_idx:
        seq = [[i for i in s if i != oov_idx] for s in seq]

    if not seq or not seq[0]:
        return None, None, "Kata tidak dikenali.", 400

    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(padded)[0]

    guard = evaluate_positive_guard(clean_text)
    print(
        f"POS_GUARD | pos={guard['pos_count']} neg={guard['neg_count']} "
        f"negInd={guard['negind_count']} negNearPos={guard['negation_near_pos']} "
        f"-> isPositive={guard['is_positive_context']}"
    )

    final_idx, positive_guard_applied = decide_class(probs, guard["is_positive_context"])
    label = CLASSES[final_idx]
    suggestion = get_suggestion(label)

    prob_percent = {
        "Normal": f"{probs[0]*100:.1f}%",
        "Moderate": f"{probs[1]*100:.1f}%",
        "Severe": f"{probs[2]*100:.1f}%",
    }

    prediction_result = {
        "label": label,
        "probs": prob_percent,
        "max_prob": f"{probs[final_idx]*100:.1f}%",
        "positive_guard_applied": positive_guard_applied,
    }
    return prediction_result, suggestion, None, 200


# --- ROUTE UTAMA & API ---
@app.route("/health", methods=["GET"])
def health():
    """Return server and model readiness status."""
    ready = model is not None and tokenizer is not None
    return jsonify({"status": "ok" if ready else "degraded", "model_loaded": ready}), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction endpoint accepting {'text': '...'}."""
    payload = request.get_json(silent=True) or {}
    raw_text = payload.get("text", "")
    if not isinstance(raw_text, str):
        return jsonify({"error": "Field 'text' harus berupa string."}), 400

    result, suggestion, error, status_code = predict_text(raw_text)
    if error:
        return jsonify({"error": error}), status_code

    return jsonify({"result": result, "suggestion": suggestion}), 200


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    original_text = ""
    suggestion = None

    if request.method == "POST":
        original_text = request.form.get("tweet_text", "")
        prediction_result, suggestion, error, _ = predict_text(original_text)
        if error:
            display_text = original_text[:MAX_INPUT_LENGTH] if len(original_text.strip()) > MAX_INPUT_LENGTH else original_text
            return render_template("index.html", error=error, original=display_text)

    return render_template(
        "index.html",
        result=prediction_result,
        suggestion=suggestion,
        original=original_text,
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", os.getenv("DEBUG", "False")).lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)
