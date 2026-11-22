# 🧠 Mental Health Check: Indonesian Twitter Depression Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![NLP](https://img.shields.io/badge/NLP-Bahasa%20Indonesia-green)

## 📌 Overview

An end-to-end AI-powered web application designed to classify depression risk levels (**Normal**, **Moderate**, **Severe**) based on Indonesian social media text (Twitter/X). This project addresses the linguistic challenges of Indonesian slang and implicit emotional context to provide an early mental health screening research prototype.

## ⚙️ Architecture & Methodology

- **Text Preprocessing (`src/preprocessing.py` & `src/slang_dict.json`):** Custom slang normalization and whitelist-based stopword removal to retain crucial negation words and implicit emotional markers.
- **Feature Selection:** TF-IDF (Term Frequency-Inverse Document Frequency) vocabulary filtration.
- **Deep Learning Model (TF-IDF + Stacked LSTM — Deployed Artifact):** The active model saved in `models/depression_model_multiclass.h5` (trained in `notebooks/train_model.ipynb`) uses a 2-layer Stacked LSTM architecture (`Embedding` → `SpatialDropout1D` → `LSTM(128)` → `LSTM(64)` → `GlobalMaxPooling1D` → `Dense(64)` with L2 regularization → `Dropout(0.5)` → `Dense(3, softmax)`), achieving **98.15% accuracy** across the 1,514-sample evaluation set shown below. *(In the broader thesis comparison, a **TF-IDF + GRU** variant also achieved **98.67% accuracy** and **98.67% macro F1-score** with faster training time.)*
- **Post-Processing (Positive Guard):** A transparent heuristic layer in Flask (`app/main.py`) to reduce false positives on casual or resilient positive expressions (e.g., *"tetap semangat"*). When triggered (`pos_count >= 2`, `pos_score >= 2`, and no negation token within 2 words before a positive token), the predicted label is adjusted to **Normal** with an explicit **Adjusted by Positive Guard** indicator while still displaying the model's raw probability distribution.
  - Decision thresholds in `app/main.py`: `THRESHOLD_SEVERE = 0.35`, `THRESHOLD_MODERATE = 0.45` (Severe takes priority above `0.35`, followed by Positive Guard, then Moderate above `0.45` and greater than Normal, otherwise Normal).
  - Input limits: maximum `1000` characters per request (`MAX_INPUT_LENGTH = 1000`) and padded sequence length `100` tokens (`MAX_LEN = 100`).

## 📊 Key Results

| Architecture | Accuracy | F1-Score (Macro) | Training Time |
| :--- | :--- | :--- | :--- |
| **TF-IDF + GRU (Proposed)** | **98.67%** | **98.67%** | **~33.9 mins** |
| TF-IDF + Stacked LSTM | 97.88% | 97.87% | ~49.4 mins |

*The TF-IDF + GRU model operates ~31.4% faster than the Stacked LSTM while maintaining strong precision on detecting Suicidal Ideation.*

### Evaluation Visuals

![Confusion Matrix](reports/figures/outputCM.png)

*Confusion matrix across 1,514 evaluation samples (`Normal`: 510/525, `Moderate`: 470/480, `Severe`: 506/509; overall 98.15% accuracy).*

![Training & Validation Accuracy and Loss Curves](reports/figures/outputgrafiklossdanakurasi.png)

*Training and validation accuracy/loss curves over 8 epochs (0–7) from `notebooks/train_model.ipynb`.*

## 📁 Project Structure

```text
deteksi_depresi/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Flask inference app
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Indonesian text preprocessing pipeline
│   └── slang_dict.json         # External slang normalization dictionary
├── notebooks/                  # Data crawling, cleaning, and model training notebooks
├── tools/
│   └── label_tool/             # Internal Flask annotation helper
├── tests/
│   └── test_preprocessing.py   # Basic preprocessing unit tests (pytest)
├── data/
│   ├── raw/
│   ├── processed/
│   └── queries/
├── models/                     # Pre-trained model & tokenizer artifacts
├── reports/
│   └── figures/                # Evaluation plots and confusion matrices
├── requirements.txt
└── README.md
```

## 🚀 How to Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/Lax28/deteksi-depresi.git
   cd deteksi-depresi
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the pre-trained artifacts (`depression_model_multiclass.h5`, `tokenizer.pkl`, and `tfidf_vectorizer.pkl`) inside the `models/` directory (excluded from Git due to file size).

4. Run the Flask application from the project root:

   ```bash
   python app/main.py
   ```
   *(or `python -m app.main`)*

5. Open your browser and navigate to `http://127.0.0.1:5000/`.
   *(Optional: set `FLASK_DEBUG=True` in `.env` to enable Flask debug mode during local development.)*
   - Available JSON endpoints: `GET /health` (server & model readiness) and `POST /api/predict` (`{"text": "..."}`).

6. *(Optional)* Run preprocessing unit tests:

   ```bash
   pytest
   ```

## 🧪 Data & Training Pipeline

1. **Crawling (`notebooks/Crawl_data_twitter.ipynb`):** Collects Indonesian Twitter/X posts using keyword queries in `data/queries/` across `Normal`, `Moderate`, and `Severe` categories.
2. **Annotation (`tools/label_tool/app.py`):** Internal Flask-based labeling helper for reviewing and assigning class labels.
3. **Cleaning (`notebooks/CleaningData.ipynb`):** Applies text normalization and prepares processed datasets.
4. **Training & Evaluation (`notebooks/train_model.ipynb`):** Trains the sequence classification model and generates the evaluation figures in `reports/figures/`.

## ⚠️ Limitations & Ethical Disclaimer

- **Not a Medical Diagnosis:** This application is an academic research prototype and **is NOT a substitute for professional medical or psychological diagnosis**.
- **Scope & Bias:** The model was trained on Indonesian Twitter/X text and may be sensitive to out-of-domain phrasing, sarcasm, or unseen slang.
- **Crisis Support:** If you or someone you know is experiencing severe emotional distress or thoughts of self-harm, please seek immediate help from a licensed psychologist, psychiatrist, or local emergency services.

## 👤 Author

- **Muhammad Jovan Adiwijaya Yanuarsyah** (2025) — Undergraduate Thesis (*Skripsi*): *Deteksi Depresi TF-IDF + Stacked LSTM / GRU*.
