from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
import os
import sys

# --- [PERBAIKAN 1] IMPORT FUNGSI DARI UTILS.PY ---
# Pastikan file utils.py ada di folder yang sama dengan app.py
try:
    from utils import preprocess_indo
    print("Berhasil memuat modul preprocessing dari utils.py")
except ImportError as e:
    print(f"Error: Gagal mengimpor utils.py. Pastikan file ada. Detail: {e}")
    # Fallback darurat (Sebaiknya jangan sampai masuk sini)
    def preprocess_indo(text): return text.lower()

# --- KONFIGURASI TENSORFLOW ---
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
app = Flask(__name__)

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'depression_model_multiclass.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')

MAX_LEN = 100
CLASSES = {0: 'Normal', 1: 'Moderate', 2: 'Severe'}

# --- LOAD RESOURCES ---
print("Memuat Model & Tokenizer...")
model = None
tokenizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Model & Tokenizer berhasil dimuat.")
    except Exception as e:
        print(f"Error loading resources: {e}")
else:
    print(f"PERINGATAN: File model tidak ditemukan di {MODEL_PATH}")

# --- FUNGSI SARAN ---
def get_suggestion(label):
    if label == 'Normal':
        return {
            'text': "Kondisi mental Anda terindikasi stabil. Tetap jaga pola hidup sehat dan luangkan waktu untuk hobi.",
            'color': 'success', 'icon': 'bi-emoji-smile'
        }
    elif label == 'Moderate':
        return {
            'text': "Terdeteksi indikasi tekanan emosional. Disarankan untuk istirahat sejenak atau bercerita kepada teman.",
            'color': 'warning', 'icon': 'bi-emoji-neutral'
        }
    elif label == 'Severe':
        return {
            'text': "Terdeteksi indikasi beban emosional berat. Sangat disarankan berkonsultasi dengan profesional.",
            'color': 'danger', 'icon': 'bi-exclamation-triangle-fill'
        }
    return {}

# --- ROUTE UTAMA ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    original_text = ""
    suggestion = None
    
    if request.method == 'POST':
        original_text = request.form.get('tweet_text', '')
        
        if not original_text.strip():
             return render_template('index.html', error="Mohon masukkan teks terlebih dahulu.")

        if original_text and model and tokenizer:
            # 1. Preprocess
            clean_text = preprocess_indo(original_text)
            
            # 2. Tokenize
            seq = tokenizer.texts_to_sequences([clean_text])
            oov_idx = tokenizer.word_index.get(tokenizer.oov_token)
            if oov_idx:
                seq = [[i for i in s if i != oov_idx] for s in seq]
            
            if not seq or not seq[0]:
                return render_template('index.html', result=None, error="Kata tidak dikenali.", original=original_text)
            
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
            
            # 3. Prediksi Model
            probs = model.predict(padded)[0]
            
                        # POSITIVE GUARD (PENJAGA KELAS NORMAL) 
            # Revisi: memakai pendekatan berbasis COUNT agar lebih pasti.
            # Intuisi: jika kalimat benar-benar bernuansa positif (>= MIN_POS_HITS kata positif) dan
            # tidak ada negasi/indikator negatif yang kuat, maka hasil boleh dioverride menjadi Normal.
            positive_tokens = {
                'bahagia','senang','semangat','syukur','alhamdulillah','happy','asik','seru',
                'positif','jernih','nikmat','menikmati','tenang','lega','baik','fokus','stabil'
            }
            negation_tokens = {'tidak','ga','gak','bukan','tak','jangan','nggak','tdk'}
            negative_indicator_tokens = {
                'sedih','hampa','kesepian','nangis','putus','putusasa','depresi',
                'bunuh','mati','akhiri','pengen','mau','beban','negatif','hopeless','putusasa'
            }

            words = clean_text.split()
            pos_count = sum(1 for w in words if w in positive_tokens)
            neg_count = sum(1 for w in words if w in negation_tokens)
            negind_count = sum(1 for w in words if w in negative_indicator_tokens)

            # Negasi lokal (mis. "tidak bahagia") -> dianggap bukan konteks positif
            pos_positions = [i for i,w in enumerate(words) if w in positive_tokens]
            negation_near_pos = False
            for i in pos_positions:
                window = words[max(0, i-2):i]  # 1-2 kata sebelum kata positif
                if any(w in negation_tokens for w in window):
                    negation_near_pos = True
                    break

            MIN_POS_HITS = 2
            pos_score = pos_count - (neg_count + negind_count)

            is_positive_context = (pos_count >= MIN_POS_HITS) and (pos_score >= 2) and (not negation_near_pos)

            print(f"POS_GUARD | pos={pos_count} neg={neg_count} negInd={negind_count} "
                    f"negNearPos={negation_near_pos} -> isPositive={is_positive_context}")
# LOGIKA KEPUTUSAN FINAL
            THRESHOLD_SEVERE = 0.35 
            THRESHOLD_MODERATE = 0.45 
            
            # Prioritas 1: Severe (Tetap paling tinggi)
            if probs[2] > THRESHOLD_SEVERE:
                final_idx = 2 
            
            # Prioritas 2: Normal Jika Terdeteksi Konteks Positif (Override Moderate)
            elif is_positive_context:
                final_idx = 0
                # Opsional: Kita bisa manipulasi probabilitas untuk tampilan agar user tidak bingung
                # (Misal: Normal jadi 90%)
                probs = [0.90, 0.09, 0.01]
            
            # Prioritas 3: Moderate (Hanya jika skor tinggi & kalahkan Normal)
            elif probs[1] > THRESHOLD_MODERATE and probs[1] > probs[0]:
                final_idx = 1 
            
            # Default: Normal
            else:
                final_idx = 0 
            
            # 4. Format Hasil
            label = CLASSES[final_idx]
            suggestion = get_suggestion(label)
            
            prob_percent = {
                'Normal': f"{probs[0]*100:.1f}%",
                'Moderate': f"{probs[1]*100:.1f}%",
                'Severe': f"{probs[2]*100:.1f}%"
            }
            
            prediction_result = {
                'label': label,
                'probs': prob_percent,
                'max_prob': f"{probs[final_idx]*100:.1f}%"
            }
            
    return render_template('index.html', 
                           result=prediction_result, 
                           suggestion=suggestion,
                           original=original_text)

if __name__ == '__main__':
    app.run(debug=True)