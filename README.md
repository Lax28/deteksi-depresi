# 🧠 Mental Health Check: Indonesian Twitter Depression Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![NLP](https://img.shields.io/badge/NLP-Bahasa%20Indonesia-green)

## 📌 Overview

An end-to-end AI-powered web application designed to classify depression risk levels (Normal, Moderate, Severe) based on Indonesian social media text (Twitter/X). This project addresses the linguistic challenges of Indonesian slang and implicit context to provide a responsive early mental health screening tool.

## ⚙️ Architecture & Methodology

- **Text Preprocessing:** Custom slang normalization and whitelist-based stopword removal to retain crucial negation words and implicit emotional markers.
- **Feature Selection:** TF-IDF (Term Frequency-Inverse Document Frequency) vocabulary filtration.
- **Deep Learning Model:** Gated Recurrent Unit (GRU) with Spatial Dropout and L2 Regularization.
- **Post-Processing (Positive Guard):** A heuristic backend logic implemented in Flask to mitigate False Positives on casual/resilience phrases (e.g., "tetap harus dijalanin").

## 📊 Key Results

Extensive hyperparameter tuning and ablation studies demonstrated that the GRU architecture outperformed the Stacked LSTM baseline in both accuracy and computational efficiency.

| Architecture | Accuracy | F1-Score (Macro) | Training Time |
| :--- | :--- | :--- | :--- |
| **TF-IDF + GRU (Proposed)** | **98.67%** | **98.67%** | **~33.9 mins** |
| TF-IDF + Stacked LSTM | 97.88% | 97.87% | ~49.4 mins |

*The GRU model operates ~31.4% faster than the Stacked LSTM while maintaining highly robust precision on detecting Suicidal Ideation.*

## 🚀 How to Run Locally

1. Clone this repository:

   ```bash
   git clone [https://github.com/Lax28/deteksi-depresi-nl.git](https://github.com/USERNAME_GITHUB_KAMU/deteksi-depresi-nlp.git)```
   cd deteksi-depresi-nlp
2. Place the pre-trained models (depression_model_multiclass.h5, tokenizer.pkl, and tfidf_vectorizer.pkl) inside the models directory. (Note: Models are not included in this repo due to size limits).

3. Run the Flask application:

    ```Bash
    python app.py```
4. Open your browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
