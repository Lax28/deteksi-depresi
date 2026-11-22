import json
import re
from pathlib import Path

from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

SLANG_DICT_PATH = Path(__file__).resolve().parent / "slang_dict.json"


def load_slang_dict(path: Path = SLANG_DICT_PATH) -> dict[str, str]:
    """Load Indonesian slang normalization mapping from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


SLANG_DICT = load_slang_dict()


def get_text_processor_resources():
    """Initialize Sastrawi stopword remover (with domain whitelist/blacklist) and stemmer."""
    factory = StopWordRemoverFactory()
    stop_factory = factory.get_stop_words()

    # --- [1] WHITELIST: KATA PENTING ---
    keep_words = {
        'tidak', 'ga', 'gak', 'gk', 'tdk', 'bukan', 'tak', 'jangan', 'jgn', 'enggak',
        'mau', 'akan', 'ingin', 'pengen', 'bakal', 'niat',
        'diri', 'sendiri', 'bisa', 'masih', 'terus', 'lagi', 'selalu',
        'banget', 'bgt', 'sekali', 'sangat', 'amat', 'paling',
        'sakit', 'rasa', 'perasaan', 'butuh', 'pikiran', 'kepikiran', 'mikir',
        'nyerah', 'menyerah', 'capek', 'lelah', 'pusing',
        'mental', 'batin', 'jiwa', 'hati', 'hampa', 'kosong', 'sepi', 'beban',
        'nangis', 'menangis', 'air', 'mata',
        'tiap', 'setiap', 'tidur', 'bangun',
        'nonton', 'film', 'drakor', 'movie', 'series', 'lagu',
        'lari', 'olahraga', 'gym', 'jogging', 'jalan', 'main', 'makan', 'minum',
        'kerja', 'kuliah', 'sekolah', 'tugas', 'skripsi', 'ujian', 'revisi',
        'habis', 'abis', 'setelah', 'usai', 'baru',
        'macet', 'panas', 'hujan', 'antri',
        'senang', 'bahagia', 'semangat', 'happy', 'bersyukur', 'alhamdulillah', 'seru', 'asik',
        'susah', 'sulit', 'berat', 'mampu', 'kuat',
        'tanpa', 'sebab', 'alasan', 'jelas',
        'guna', 'berguna', 'berharga', 'arti', 'berarti',
        'ada', 'tiada', 'hilang', 'lenyap', 'pergi',
        'hidup', 'mati', 'bunuh', 'akhiri',
    }

    # --- [2] BLACKLIST: KATA SAMPAH ---
    more_stopwords = [
        'yg', 'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'buat', 'untuk', 'utk', 'karena', 'krn',
        'sama', 'sm', 'dg', 'dengan', 'spt', 'seperti', 'pas', 'saat', 'ketika', 'kalo', 'kalau', 'kl',
        'apa', 'kenapa', 'knp', 'gimana', 'gmn', 'bagaimana', 'siapa', 'tapi', 'tp', 'tpi', 'tetapi',
        'jadi', 'jd', 'jdi', 'maka', 'lalu', 'kemudian', 'sih', 'mah', 'tuh', 'dong',
        'kok', 'deh', 'donk', 'kan', 'kapan', 'kpn', 'dimana', 'dmn', 'hal',
        'ni', 'tu', 'si', 'ae', 'doang',
        'udah', 'sudah', 'sdh', 'udh', 'dah', 'blm', 'belum', 'dpt', 'banyak',
        'hari', 'tadi', 'kemarin', 'kmrn', 'besok', 'bsk',
        'dulu', 'sekarang', 'skrg', 'jam', 'waktu', 'kali', 'nanti', 'ntar',
        'aja', 'saja', 'cuma', 'cuman', 'cm', 'hanya', 'malah', 'justru', 'pasti', 'memang', 'emang',
        'bener', 'benar', 'mungkin', 'kayak', 'kyk', 'kaya', 'kek', 'seolah', 'biar', 'mari', 'yuk',
        'wkwk', 'wkwkwk', 'haha', 'hehe', 'hihi', 'wk', 'anjir', 'njir', 'anjing', 'bang', 'mas',
        'mba', 'kak', 'bro', 'sis', 'gan', 'guys', 'gaes', 'yagesya', 'wir', 'bestie', 'ges',
        'btw', 'fyi', 'otw', 'gitu', 'gt', 'gtu', 'gini', 'gn', 'karna', 'sebab', 'supaya',
        'padahal', 'pdhl', 'walau', 'meski', 'pake', 'pakai', 'bilang',
        'ngomong', 'bicara', 'tau', 'tahu', 'liat', 'lihat', 'ngeliat', 'nemu', 'ketemu', 'dapet',
        'punya', 'milik', 'semua', 'segala', 'seluruh', 'paling', 'lebih', 'kurang',
        'aku', 'saya', 'gue', 'gw', 'gua', 'kita', 'kamu', 'lo', 'lu', 'kalian', 'mereka', 'dia',
        'ku', 'mu', 'nya', 'u', 'km', 'orang', 'org', 'sender', 'nder',
    ]

    all_stopwords = stop_factory + more_stopwords
    final_stopwords = [w for w in all_stopwords if w not in keep_words]

    dictionary = ArrayDictionary(final_stopwords)
    stopword_remover = StopWordRemover(dictionary)
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    return stopword_remover, stemmer


_stopword_remover, _stemmer = get_text_processor_resources()


def preprocess_indo(text: str) -> str:
    """Normalize slang, clean URLs/mentions/punctuation, remove stopwords, and stem Indonesian text."""
    text = str(text).lower()

    # 1. Normalisasi Slang (sebelum cleaning regex agar karakter tidak hilang)
    words = text.split()
    words = [SLANG_DICT.get(word, word) for word in words]
    text = ' '.join(words)

    # 2. Cleaning Regex
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Stopword & Stemming
    text = _stopword_remover.remove(text)
    text = _stemmer.stem(text)

    return text
