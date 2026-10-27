"""Local demo only. Reads URL text and never visits the target site.
Run after the notebook writes model_bundle.joblib into this directory:
    python app.py
"""
from __future__ import annotations

from pathlib import Path
import os
import threading

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURE_VERSION, extract_lexical_features

APP_DIR = Path(__file__).resolve().parent
BUNDLE_FILE = APP_DIR / "model_bundle.joblib"
if not BUNDLE_FILE.is_file():
    raise RuntimeError("model_bundle.joblib is missing. Run the notebook export cell first.")

# Only load trusted joblib files you trained yourself. joblib/pickle executes code.
bundle = joblib.load(BUNDLE_FILE)
if bundle["feature_version"] != FEATURE_VERSION:
    raise RuntimeError("Feature extractor version does not match trained model!")

MODEL = bundle["model"]
COLUMNS = list(bundle["feature_names"])
THRESHOLD = float(bundle["threshold"])
BACKGROUND = np.asarray(bundle["shap_background"], dtype=float)
INDEX_PHISH = list(MODEL.classes_).index(0)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
_shap_lock = threading.Lock()
_explainer = None


def _score_numpy(a):
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    frame = pd.DataFrame(arr, columns=COLUMNS)
    return MODEL.predict_proba(frame)[:, INDEX_PHISH]


def _frame_for_url(url):
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    features = extract_lexical_features(url)
    return pd.DataFrame([{key: features[key] for key in COLUMNS}], columns=COLUMNS)


def predict_url(url: str) -> dict:
    features = _frame_for_url(url)
    score = float(MODEL.predict_proba(features)[0, INDEX_PHISH])
    # Abstain around validation-selected threshold; neither side guarantees safety.
    review_band = min(0.08, THRESHOLD / 2, (1 - THRESHOLD) / 2)
    if score >= THRESHOLD + review_band:
        decision = "LIKELY PHISHING"
        label = 0
    elif score <= THRESHOLD - review_band:
        decision = "LIKELY LEGITIMATE"
        label = 1
    else:
        decision = "INSPECT"
        label = None
    return {
        "decision": decision,
        "predicted_label": label,
        "phishing_score": round(score, 6),
        "threshold": round(THRESHOLD, 4),
        "model_name": bundle["model_name"],
        "warning": "URL-only laboratory estimate. Do not treat this as proof that a site is safe.",
    }


def explain_url(url: str) -> list[dict]:
    import shap
    global _explainer
    if _explainer is None:
        with _shap_lock:
            if _explainer is None:
                _explainer = shap.Explainer(
                    _score_numpy, BACKGROUND,
                    algorithm="permutation", feature_names=COLUMNS
                )
    x = _frame_for_url(url)
    # Using one permutation pass to make on-demand explanations practical.
    values = _explainer(x.to_numpy(dtype=float), max_evals=2 * len(COLUMNS) + 1)
    impacts = values.values[0]
    ranking = np.argsort(np.abs(impacts))[::-1][:6]
    return [
        {
            "feature": COLUMNS[int(i)],
            "value": round(float(x.iloc[0, int(i)]), 4),
            "impact_on_phishing_score": round(float(impacts[i]), 5),
            "direction": "raises risk" if impacts[i] > 0 else "lowers risk",
        }
        for i in ranking
    ]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        url = body.get("url", "")
        outcome = predict_url(url)
        if body.get("explain") is True:
            outcome["explanation"] = explain_url(url)
        return jsonify(outcome)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected prediction failure")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


if __name__ == "__main__":
    # Demo: bind to localhost, disable Flask's debug server.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
