import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
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
CLASSES = {0: "Normal", 1: "Moderate", 2: "Severe"}

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


# --- ROUTE UTAMA ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    original_text = ""
    suggestion = None

    if request.method == "POST":
        original_text = request.form.get("tweet_text", "")
        stripped_text = original_text.strip()

        if not stripped_text:
            return render_template("index.html", error="Mohon masukkan teks terlebih dahulu.")

        if len(stripped_text) > MAX_INPUT_LENGTH:
            return render_template(
                "index.html",
                error=f"Teks terlalu panjang (maksimal {MAX_INPUT_LENGTH} karakter).",
                original=original_text[:MAX_INPUT_LENGTH],
            )

        if model is None or tokenizer is None:
            return render_template(
                "index.html",
                error="Model atau tokenizer belum tersedia di server.",
                original=original_text,
            )

        # 1. Preprocess
        clean_text = preprocess_indo(original_text)

        # 2. Tokenize
        seq = tokenizer.texts_to_sequences([clean_text])
        oov_idx = tokenizer.word_index.get(tokenizer.oov_token)
        if oov_idx:
            seq = [[i for i in s if i != oov_idx] for s in seq]

        if not seq or not seq[0]:
            return render_template(
                "index.html",
                result=None,
                error="Kata tidak dikenali.",
                original=original_text,
            )

        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        # 3. Prediksi Model
        probs = model.predict(padded)[0]

        # POSITIVE GUARD (PENJAGA KELAS NORMAL)
        positive_tokens = {
            "bahagia", "senang", "semangat", "syukur", "alhamdulillah", "happy", "asik", "seru",
            "positif", "jernih", "nikmat", "menikmati", "tenang", "lega", "baik", "fokus", "stabil",
        }
        negation_tokens = {"tidak", "ga", "gak", "bukan", "tak", "jangan", "nggak", "tdk"}
        negative_indicator_tokens = {
            "sedih", "hampa", "kesepian", "nangis", "putus", "putusasa", "depresi",
            "bunuh", "mati", "akhiri", "pengen", "mau", "beban", "negatif", "hopeless",
        }

        words = clean_text.split()
        pos_count = sum(1 for w in words if w in positive_tokens)
        neg_count = sum(1 for w in words if w in negation_tokens)
        negind_count = sum(1 for w in words if w in negative_indicator_tokens)

        pos_positions = [i for i, w in enumerate(words) if w in positive_tokens]
        negation_near_pos = False
        for i in pos_positions:
            window = words[max(0, i - 2) : i]
            if any(w in negation_tokens for w in window):
                negation_near_pos = True
                break

        MIN_POS_HITS = 2
        pos_score = pos_count - (neg_count + negind_count)
        is_positive_context = (pos_count >= MIN_POS_HITS) and (pos_score >= 2) and (not negation_near_pos)

        print(
            f"POS_GUARD | pos={pos_count} neg={neg_count} negInd={negind_count} "
            f"negNearPos={negation_near_pos} -> isPositive={is_positive_context}"
        )

        # LOGIKA KEPUTUSAN FINAL
        THRESHOLD_SEVERE = 0.35
        THRESHOLD_MODERATE = 0.45
        positive_guard_applied = False

        if probs[2] > THRESHOLD_SEVERE:
            final_idx = 2
        elif is_positive_context:
            final_idx = 0
            positive_guard_applied = True
        elif probs[1] > THRESHOLD_MODERATE and probs[1] > probs[0]:
            final_idx = 1
        else:
            final_idx = 0

        # 4. Format Hasil
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

    return render_template(
        "index.html",
        result=prediction_result,
        suggestion=suggestion,
        original=original_text,
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", os.getenv("DEBUG", "False")).lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)
