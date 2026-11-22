import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app, decide_class, evaluate_positive_guard, get_suggestion


def test_positive_guard_triggers_on_clear_positive_text():
    guard = evaluate_positive_guard("hari ini senang bahagia syukur")
    assert guard["pos_count"] >= 2
    assert guard["negation_near_pos"] is False
    assert guard["is_positive_context"] is True


def test_positive_guard_blocked_by_nearby_negation():
    guard = evaluate_positive_guard("tidak bahagia tidak tenang")
    assert guard["negation_near_pos"] is True
    assert guard["is_positive_context"] is False


def test_decide_class_thresholds_and_priority():
    # Severe threshold (> 0.35) overrides positive guard
    assert decide_class([0.30, 0.30, 0.40], True) == (2, False)
    # Positive guard overrides Moderate when Severe <= 0.35
    assert decide_class([0.40, 0.50, 0.10], True) == (0, True)
    # Moderate threshold (> 0.45 and > Normal)
    assert decide_class([0.35, 0.55, 0.10], False) == (1, False)
    # Fallback to Normal
    assert decide_class([0.50, 0.40, 0.10], False) == (0, False)


def test_get_suggestion_labels():
    assert get_suggestion("Normal")["color"] == "success"
    assert get_suggestion("Moderate")["color"] == "warning"
    assert get_suggestion("Severe")["color"] == "danger"
    assert get_suggestion("Unknown") == {}


def test_health_and_api_predict_validation():
    client = app.test_client()
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["status"] in ("ok", "degraded")

    empty_resp = client.post("/api/predict", json={"text": "   "})
    assert empty_resp.status_code == 400

    long_resp = client.post("/api/predict", json={"text": "a" * 1001})
    assert long_resp.status_code == 400
