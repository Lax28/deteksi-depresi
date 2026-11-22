import sys
from pathlib import Path

# Ensure project root is importable when running pytest from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_slang_dict, preprocess_indo


def test_slang_dict_loaded():
    slang = load_slang_dict()
    assert slang["bgt"] == "banget"
    assert slang["gak"] == "tidak"
    assert slang["mager"] == "malas gerak"


def test_slang_normalization():
    result = preprocess_indo("capek bgt hari ini mager")
    tokens = result.split()
    assert "banget" in tokens
    assert "malas" in tokens
    assert "gerak" in tokens
    assert "capek" in tokens


def test_negation_words_retained():
    for neg in ("tidak", "bukan", "jangan"):
        result = preprocess_indo(f"{neg} bahagia")
        tokens = result.split()
        assert neg in tokens, f"Kata negasi '{neg}' seharusnya tidak terhapus: {result}"


def test_slang_negation_normalized_and_retained():
    result = preprocess_indo("gw gak bahagia bgt")
    tokens = result.split()
    assert "tidak" in tokens
    assert "banget" in tokens
    assert "bahagia" in tokens


def test_url_mention_hashtag_cleaned():
    result = preprocess_indo("Sedih bgt @teman #curhat https://example.com")
    assert "http" not in result
    assert "example" not in result
    assert "teman" not in result
    assert "curhat" not in result
    assert "sedih" in result
    assert "banget" in result
