# URL-only phishing detection — reproducible academic lab

**Important:** PhiUSIIL's legitimate URL examples have severe path/`www.` distribution bias. The notebook deliberately *does not use* path, query, scheme, or `www.` as features. That makes the classifier robust to superficial changes in these components but unable to catch phishing that resides solely on the path of a benign-looking hostname. **No classifier trained on PhiUSIIL alone should be described as real-world reliable without an independent, representative benign-and-phishing benchmark.**

## Colab workflow

1. Open `phishing_detection_lab.ipynb` in Google Colab.
2. Runtime → Run all. The first cell installs all required libraries. Internet is needed to fetch UCI 967.
3. The notebook: audits data → removes duplicate URLs → domain-disjoint train/validation/test → builds the exact features defined in `features.py` → training-only feature selection → fits Logistic Regression, Random Forest, XGBoost and soft-voting → selects a **provisional** champion by validation F1 under an FPR constraint → plots ROC/PR curves → reports untouched test-set metrics → produces SHAP explanations and counterfactual invariance checks.
4. Final cell creates `model_bundle.joblib` and packages `phishing_flask_project.zip` (contains Flask app, frontend, feature extraction and model). Download that archive from Colab.
5. Optionally place a fresh, independently verified `external_urls.csv` next to the notebook with columns `URL,label` (0=phishing, 1=legitimate). The optional notebook cell evaluates it without fitting on it. **Do not reuse the old 145-URL phishing sample as an untouched final test.**

## Flask (local)

Unzip `phishing_flask_project.zip`, then:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000/.

For POST integration:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.org/login","explain":true}'
```

Example JSON fields: `decision` (`LIKELY PHISHING`, `INSPECT`, or `LIKELY LEGITIMATE`), `predicted_label` (`0`, `null`, or `1`), `phishing_score`, and optional `explanation` (model-agnostic SHAP impact). **The score is not validated as a real-world probability.**

The server never follows or visits submitted URLs. It rejects empty/non-HTTP URLs, caps input size, uses a fixed feature schema, and binds to localhost by default. This is a demo, not a hardened public service. Add authentication, rate limits, HTTPS, monitoring, and proper deployment configuration before exposing it publicly. SHAP is computationally expensive and should be rate-limited in a public app.

## Feature policy

`features.py` is the **single source of truth** used during training, testing, SHAP and Flask inference. It uses only hostname and registered-domain string metrics. `URLSimilarityIndex`, HTML features, path/query lengths and `www.` are excluded on purpose. `tldextract` uses its bundled PSL snapshot offline for consistent features; update dependencies cautiously and retrain if definitions change.

## Model selection & limitations

- 0 = phishing, 1 = legitimate throughout. ROC-AUC and PR-AUC use `(label==0)` as the positive class, and `predict_proba` is indexed through the model's `classes_`.
- Only the development split is used for mutual information and correlation pruning. The validation split chooses a decision threshold and model; the test split is touched **once** for final performance reporting.
- Groups are registered domains, to reduce leakage between related URLs. The splits are not guaranteed to have identical class proportions.
- Validation target: a false-alarm rate no greater than 5%, then maximize phishing F1. This threshold is only a laboratory operating point, not a real-world guarantee.
- A low-risk answer is **not** verification that a site is safe; lexical-only detection misses many phishing websites, particularly short simple hostnames and attacks on compromised legitimate domains.

## Basic tests

```bash
python -m unittest discover -s tests
```
